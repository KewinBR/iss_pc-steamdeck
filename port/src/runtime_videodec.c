/* libSceVideodec: H.264 access units decoded by FFmpeg into the game's NV12 frame buffers.
 * Structures, sizes and the resource estimate follow shadPS4's videodec (GPL-2.0-or-later);
 * pictures larger than the game's frame buffer are an error code instead of an overflow. */
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

#define ERR_API_FAIL ((int32_t)0x80C10000)
#define ERR_STRUCT_SIZE ((int32_t)0x80C10002)
#define ERR_HANDLE ((int32_t)0x80C10003)
#define ERR_AU_SIZE ((int32_t)0x80C10009)
#define ERR_AU_POINTER ((int32_t)0x80C1000A)
#define ERR_FRAME_BUFFER_SIZE ((int32_t)0x80C1000B)
#define ERR_FRAME_BUFFER_POINTER ((int32_t)0x80C1000C)
#define ERR_ARGUMENT_POINTER ((int32_t)0x80C1000F)

typedef struct {
    uint64_t this_size;
    uint32_t codec_type, profile, max_level;
    int32_t max_frame_width, max_frame_height, max_dpb_frame_count;
    uint64_t videodec_flags;
} ConfigInfo;
typedef struct {
    uint64_t this_size, cpu_memory_size;
    void *cpu_memory;
    uint64_t cpu_gpu_memory_size;
    void *cpu_gpu_memory;
    uint64_t max_frame_buffer_size;
    uint32_t frame_buffer_alignment;
} ResourceInfo;
typedef struct { uint64_t this_size; void *handle; uint64_t version; } Ctrl;
typedef struct { uint64_t this_size; void *frame_buffer; uint64_t frame_buffer_size; } FrameBuffer;
typedef struct {
    uint32_t num_units_in_tick, time_scale;
    uint8_t fixed_frame_rate_flag, aspect_ratio_idc;
    uint16_t sar_width, sar_height;
    uint8_t colour_primaries, transfer_characteristics, matrix_coefficients, video_full_range_flag;
    uint32_t crop_left, crop_right, crop_top, crop_bottom;
} AvcInfo;
typedef struct {
    uint64_t this_size;
    uint32_t is_valid, codec_type, frame_width, frame_pitch, frame_height, is_error_pic;
    uint64_t pts_data, attached_data;
    union { uint8_t reserved[64]; AvcInfo avc; } codec;
} PictureInfo;
typedef struct { uint64_t this_size; void *au_data; uint64_t au_size, pts_data, dts_data, attached_data; } InputData;
_Static_assert(sizeof(ConfigInfo)==40 && sizeof(ResourceInfo)==56 && sizeof(Ctrl)==24 && sizeof(FrameBuffer)==24 &&
               sizeof(PictureInfo)==112 && sizeof(InputData)==48,"Videodec guest layouts");

typedef struct { AVCodecContext *codec; AVPacket *packet; AVFrame *frame, *nv12; struct SwsContext *sws; } Decoder;

static uint32_t align(uint32_t value, uint32_t to) { return (value+to-1)&~(to-1); }

static ABI int32_t videodec_query(const ConfigInfo *config, ResourceInfo *out) {
    if (!config || !out) return ERR_ARGUMENT_POINTER;
    if (config->this_size!=sizeof(*config) || out->this_size!=sizeof(*out)) return ERR_STRUCT_SIZE;
    int32_t width=config->max_frame_width, height=config->max_frame_height;
    if (width<=0 || height<=0) { width=config->max_level>=150 ? 3840 : 1920; height=config->max_level>=150 ? 2160 : 1080; }
    uint64_t frame=(uint64_t)align((uint32_t)width,256)*align((uint32_t)height,16)*3/2;
    uint64_t padded=((frame+255)&~UINT64_C(255))+0x4000;
    uint64_t surfaces=(uint64_t)(config->max_dpb_frame_count>0 ? config->max_dpb_frame_count : 8)+2;
    out->cpu_memory=NULL; out->cpu_gpu_memory=NULL;
    out->cpu_memory_size=16u<<20;
    out->cpu_gpu_memory_size=padded*surfaces+(8u<<20);
    out->max_frame_buffer_size=padded;
    out->frame_buffer_alignment=0x100;
    return 0;
}
static ABI int32_t videodec_create(const ConfigInfo *config, const ResourceInfo *resource, Ctrl *out) {
    if (!config || !resource || !out) return ERR_ARGUMENT_POINTER;
    if (config->this_size!=sizeof(*config) || resource->this_size!=sizeof(*resource)) return ERR_STRUCT_SIZE;
    const AVCodec *codec=avcodec_find_decoder(AV_CODEC_ID_H264);
    Decoder *d=calloc(1,sizeof(*d));
    if (!codec || !d) { free(d); return ERR_API_FAIL; }
    d->codec=avcodec_alloc_context3(codec);
    d->packet=av_packet_alloc();
    d->frame=av_frame_alloc();
    d->nv12=av_frame_alloc();
    if (d->codec) d->codec->flags|=AV_CODEC_FLAG_COPY_OPAQUE; /* attached data follows its picture */
    if (!d->codec || !d->packet || !d->frame || !d->nv12 || avcodec_open2(d->codec,codec,NULL)<0) {
        avcodec_free_context(&d->codec); av_packet_free(&d->packet); av_frame_free(&d->frame); av_frame_free(&d->nv12);
        free(d); return ERR_API_FAIL;
    }
    out->this_size=sizeof(*out); out->handle=d; out->version=1;
    return 0;
}

/* steam-deck: full-screen movies (art/movies/cutscene: the Sucker Punch intro, cutscenes) in 16:10. With the
 * game's virtual screen at 16:10 (TLG_VIDEO_CROP_1610=1, set by the launch script for the 16:10 resolutions)
 * a 16:9 picture is handed over cropped at the centre to 16:10 (1920x1080 -> 1728x1080: 96 px off each side);
 * the game fits it to the screen, so it fills it with no bars and no stretching. Every movie is cropped (power
 * previews and in-game TVs too: the Kewin chose immersion); movie_path is kept for the log. */
static char movie_path[256];
/* steam-deck: pular video (segurar o botao de menu). O .bsf aberto agora, a ultima leitura de outro arquivo (o jogo carregando
 * por tras), o ultimo quadro decodificado e o pedido de pular, que faz o .bsf ler fim de arquivo. */
#include <time.h>
static volatile int movie_fd=-1, movie_skip=0;
static volatile uint64_t movie_closed_at;
static volatile uint64_t movie_last_other_read, movie_last_decode;
static uint64_t movie_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000+(uint64_t)(t.tv_nsec/1000000); }
void tlggpu_movie_skip_ui(int state, float progress) __attribute__((weak));
static uint64_t movie_first_read, movie_bytes, movie_bytes_3s;
void runtime_audio_mute_for(uint64_t ms) __attribute__((weak));
static char movie_cut_path[256]; /* o video da cena (movie_path e o ultimo .bsf aberto, pode ser uma TV) */
static volatile uint64_t movie_last_read, movie_mute_min;
void runtime_movie_opened(int fd) {
    snprintf(movie_cut_path,sizeof(movie_cut_path),"%s",movie_path);
    /* Um pulo anterior que nao chegou ao fim (o jogo nao leu o controle no carregamento): o audio volta. */
    if (movie_skip && runtime_audio_mute_for) runtime_audio_mute_for(0);
    movie_closed_at=0; movie_fd=fd; movie_skip=0; movie_first_read=0; movie_bytes=0; movie_bytes_3s=0; }
static volatile uint64_t movie_skip_at;
void runtime_movie_closed(int fd) {
    if (fd!=movie_fd) return;
    if (movie_skip) {
        printf("Videodec: video fechado %llu ms depois do pulo\n",(unsigned long long)(movie_now_ms()-movie_skip_at));
        fflush(stdout);
    }
    if (movie_skip) movie_closed_at=movie_now_ms();
    movie_fd=-1; /* o pulo segue (movie_skip) ate 1,5 s depois de fechar ou o jogo voltar a desenhar a cena */
}
uint64_t tlggpu_frame_draws(void) __attribute__((weak));
int runtime_movie_eof(int fd) { return movie_skip && fd>=0 && fd==movie_fd; }
uint64_t tlggpu_frame_draws(void) __attribute__((weak));
/* Leitura pesada de outro arquivo = carregando: mais de 2 MB em 1,5 s (o audio que toca junto le pouco). */
void runtime_movie_note_read(int fd, uint64_t size) {
    static uint64_t window_start, bytes;
    if (fd==movie_fd && movie_fd>=0) { /* quanto do video o jogo ja leu (o audio adiantado vem junto) */
        const uint64_t t=movie_now_ms();
        if (!movie_first_read) movie_first_read=t;
        if (t-movie_first_read<3000) movie_bytes_3s+=size;
        movie_bytes+=size;
        movie_last_read=t;
        return;
    }
    if (movie_fd<0) return;
    const uint64_t now=movie_now_ms();
    if (now-window_start>1500) { window_start=now; bytes=0; }
    bytes+=size;
    if (bytes>(2u<<20)) movie_last_other_read=now;
}
/* Cada leitura do controle, com o botao de menu. Estado da tela: 0 nada, 1 carregando, 2 "Pular" (progress = quanto
 * do anel), 3 pulando (coberta). Volta 1 se o botao fica com o pular nesta leitura, 2 se foi um toque rapido (menos
 * de 0,3 s: o jogo recebe o toque e o pause funciona durante o video). */
int runtime_movie_poll(int a_held) {
    static int engaged=0, shown=-1, owned=0, skipped=0;
    static uint64_t hold_since=0, owned_at=0;
    const uint64_t now=movie_now_ms();
    /* Tocando: a cena aberta e sendo lida, quadro novo e o jogo sem desenhar a cena 3D (o video cobre a tela: 1 a 3
     * desenhos por quadro; o jogo le o video a cada 4 s no maximo). As TVs, a interface e um .bsf de cena aberto durante o jogo nao contam. */
    const int playing=movie_fd>=0 && now-movie_last_decode<1500 && movie_last_read && now-movie_last_read<8000 &&
                      (!tlggpu_frame_draws || tlggpu_frame_draws()<100);
    int ret=0;
    { static int was; /* o aperto com uma cena aberta vai pro log: o que fez contar como video tocando ou nao */
        if (a_held && !was && movie_fd>=0) {
            printf("Videodec: menu apertado: decodificado ha %llu ms, lido ha %llu ms, %llu desenhos, tocando %d\n",
                   (unsigned long long)(now-movie_last_decode),(unsigned long long)(movie_last_read ? now-movie_last_read : 0),
                   (unsigned long long)(tlggpu_frame_draws ? tlggpu_frame_draws() : 0),playing);
            fflush(stdout);
        }
        was=a_held;
    }
    if (a_held && !owned && playing && !movie_skip) { owned=1; skipped=0; owned_at=now; }
    if (owned) {
        if (a_held) ret=1;
        else { if (!skipped && now-owned_at<300) ret=2; owned=0; }
    }
    int state=0; float progress=0;
    if (movie_skip) { /* pulada: coberta ate o jogo voltar a desenhar a cena 3D (teto de 90 s) */
        static uint64_t busy_since, last_log;
        const uint64_t draws=tlggpu_frame_draws ? tlggpu_frame_draws() : 0;
        if (now-last_log>=1000) {
            last_log=now;
            printf("Videodec: pulando, %llu desenhos por quadro\n",(unsigned long long)draws);
            fflush(stdout);
        }
        if (draws>=300) { if (!busy_since) busy_since=now; } else busy_since=0;
        state=3;
        /* Fim: o jogo desenhando a cena 3D por 1,5 s, ou 1,5 s depois de fechar o video (a abertura volta pro menu,
         * que desenha pouco), ou o teto de 90 s. */
        if ((busy_since && now-busy_since>=1500 && now-movie_skip_at>=1000) ||
            (movie_closed_at && now-movie_closed_at>=1500) || now-movie_skip_at>90000) {
            printf("Videodec: cena encerrada %llu ms depois do pulo\n",(unsigned long long)(now-movie_skip_at));
            fflush(stdout);
            movie_skip=0; busy_since=0; movie_closed_at=0; state=0;
            /* O audio do video que o jogo ja tinha lido toca ~6,5 s depois do pulo (medido na viatura): o mudo dura
             * pelo menos ate la, mesmo com a cena de volta (8 s, ou 60% da estimativa, ate 12 s: o Kewin ainda ouviu
             * o fim do audio em alguns pulos). */
            const uint64_t since=now-movie_skip_at;
            if (runtime_audio_mute_for) runtime_audio_mute_for(since<movie_mute_min ? movie_mute_min-since : 0);
        }
        engaged=0; hold_since=0;
    } else if (!playing || !owned) { engaged=0; hold_since=0; }
    else {
        if (a_held && !engaged) engaged=1;
        if (engaged) {
            const int loading=now-movie_last_other_read<1500;
            if (loading) { state=1; hold_since=0; }
            else {
                state=2;
                if (a_held) { if (!hold_since) hold_since=now; progress=(float)(now-hold_since)/1000.0f; }
                else hold_since=0;
                if (progress>=1.0f) {
                    movie_skip=1; engaged=0; skipped=1;
                    /* O audio ja lido: bytes lidos / taxa constante (depois dos 3 s do adiantamento inicial) - tocado. */
                    const uint64_t played=now-movie_first_read;
                    uint64_t mute=6500;
                    if (movie_first_read && played>6000 && movie_bytes>movie_bytes_3s) {
                        const double rate=(double)(movie_bytes-movie_bytes_3s)/(double)(played-3000); /* bytes/ms */
                        const double ahead=(double)movie_bytes/rate-(double)played;
                        mute=ahead<1000 ? 1000 : ahead>15000 ? 15000 : (uint64_t)ahead;
                    }
                    /* Medido na cena da viatura (23/10): a estimativa deu 12,3 s e o audio da cena acabou em ~6,5 s (o buffer
                     * de audio do jogo e menor que o de video). Fica a estimativa no log e o mudo em ate 7 s. */
                    const uint64_t estimate=mute;
                    movie_mute_min=estimate*6/10<8000 ? 8000 : estimate*6/10>12000 ? 12000 : estimate*6/10;
                    mute=90000; /* ate o jogo fechar o video (runtime_movie_closed) */
                    movie_skip_at=now; state=3;
                    if (runtime_audio_mute_for) runtime_audio_mute_for(mute);
                    printf("Videodec: audio da cena mudo por %llu ms\n",(unsigned long long)mute);
                    printf("Videodec: video pulado (%s) t=%llu ms, estimativa do audio %llu ms\n",movie_cut_path,
                           (unsigned long long)now,(unsigned long long)estimate);
                    fflush(stdout);
                }
            }
        }
    }
    if (tlggpu_movie_skip_ui && (state!=shown || state==2)) { tlggpu_movie_skip_ui(state,progress>1.0f ? 1.0f : progress); shown=state; }
    return ret;
}
void runtime_videodec_note_open(const char *guest) {
    const size_t n=guest ? strlen(guest) : 0;
    if (n>4 && !strcmp(guest+n-4,".bsf")) {
        snprintf(movie_path,sizeof(movie_path),"%s",guest);
        /* steam-deck: cada video aberto vai pro log (lista dos videos que o "pular video" vai cobrir). */
        printf("Videodec: video %s\n",guest);
        fflush(stdout);
    }
}
static int crop_1610(void) {
    static int on=-1;
    if (on<0) { const char *e=getenv("TLG_VIDEO_CROP_1610"); on=e && e[0]=='1'; }
    return on; /* todos os vídeos (pedido do Kewin: mais imersão), não só os de tela cheia */
}

/* The decoded picture into the game's buffer: luma, then interleaved chroma, both with a pitch
 * of the width aligned to 64 and the height aligned to 16 (the last rows repeated). */
static int32_t picture(Decoder *d, FrameBuffer *buffer, PictureInfo *info) {
    AVFrame *f=d->frame;
    if (f->format!=AV_PIX_FMT_NV12) {
        av_frame_unref(d->nv12);
        d->nv12->format=AV_PIX_FMT_NV12; d->nv12->width=f->width; d->nv12->height=f->height;
        d->sws=sws_getCachedContext(d->sws,f->width,f->height,(enum AVPixelFormat)f->format,
                                    f->width,f->height,AV_PIX_FMT_NV12,SWS_FAST_BILINEAR,NULL,NULL,NULL);
        if (!d->sws || av_frame_get_buffer(d->nv12,0)<0 ||
            sws_scale(d->sws,(const uint8_t *const *)f->data,f->linesize,0,f->height,d->nv12->data,d->nv12->linesize)<0)
            return ERR_API_FAIL;
        f=d->nv12;
    }
    uint32_t width=(uint32_t)f->width, height=(uint32_t)f->height, x0=0;
    if (crop_1610() && (uint64_t)width*10>(uint64_t)height*16) {
        const uint32_t cw=(height*16/10)&~1u;
        x0=((width-cw)/2)&~1u; /* par: no NV12 o deslocamento em bytes vale pro croma também */
        width=cw;
        static int logged;
        if (!logged) { logged=1; printf("Videodec: full-screen movie cropped to 16:10 (%ux%u, %s)\n",width,height,movie_path); }
    }
    uint32_t pitch=align(width,64), rows=align(height,16);
    if ((uint64_t)pitch*rows*3/2>buffer->frame_buffer_size) return ERR_FRAME_BUFFER_SIZE;
    unsigned char *luma=buffer->frame_buffer, *chroma=luma+(size_t)pitch*rows;
    for (uint32_t y=0;y<rows;++y) {
        uint32_t from=y<height ? y : height-1;
        memcpy(luma+(size_t)y*pitch,f->data[0]+(size_t)from*(size_t)f->linesize[0]+x0,width);
        if (y<rows/2) {
            uint32_t c=y<height/2 ? y : height/2-1;
            memcpy(chroma+(size_t)y*pitch,f->data[1]+(size_t)c*(size_t)f->linesize[1]+x0,width);
        }
    }
    info->is_valid=1; info->codec_type=0;
    info->frame_width=align(width,16); info->frame_pitch=pitch; info->frame_height=rows;
    info->is_error_pic=0;
    info->pts_data=(uint64_t)d->frame->pts;
    info->attached_data=(uint64_t)(uintptr_t)d->frame->opaque;
    memset(&info->codec,0,sizeof(info->codec));
    info->codec.avc.colour_primaries=(uint8_t)d->frame->color_primaries;
    info->codec.avc.crop_right=pitch-width;
    info->codec.avc.crop_bottom=rows-height;
    return 0;
}
static int32_t receive(Decoder *d, FrameBuffer *buffer, PictureInfo *info) {
    av_frame_unref(d->frame);
    int r=avcodec_receive_frame(d->codec,d->frame);
    if (r==AVERROR(EAGAIN) || r==AVERROR_EOF) return 0;
    return r<0 ? ERR_API_FAIL : picture(d,buffer,info);
}
static int32_t check(Ctrl *ctrl, FrameBuffer *buffer, PictureInfo *info) {
    if (!ctrl || !buffer || !info) return ERR_ARGUMENT_POINTER;
    if (ctrl->this_size!=sizeof(*ctrl) || buffer->this_size!=sizeof(*buffer) || info->this_size!=sizeof(*info)) return ERR_STRUCT_SIZE;
    if (!ctrl->handle) return ERR_HANDLE;
    if (!buffer->frame_buffer) return ERR_FRAME_BUFFER_POINTER;
    info->is_valid=0;
    return 0;
}
static ABI int32_t videodec_decode(Ctrl *ctrl, const InputData *input, FrameBuffer *buffer, PictureInfo *info) {
    if (!input) return ERR_ARGUMENT_POINTER;
    int32_t r=check(ctrl,buffer,info);
    if (r) return r;
    if (!input->au_data) return ERR_AU_POINTER;
    if (!input->au_size || input->au_size>INT32_MAX) return ERR_AU_SIZE;
    Decoder *d=ctrl->handle;
    d->packet->data=input->au_data; d->packet->size=(int)input->au_size;
    d->packet->pts=(int64_t)input->pts_data; d->packet->dts=(int64_t)input->dts_data;
    d->packet->opaque=(void *)(uintptr_t)input->attached_data;
    movie_last_decode=movie_now_ms(); /* steam-deck: um video esta tocando (pular video) */
    int sent=avcodec_send_packet(d->codec,d->packet);
    if (sent==AVERROR_EOF) { avcodec_flush_buffers(d->codec); sent=avcodec_send_packet(d->codec,d->packet); } /* after a flush */
    d->packet->data=NULL; d->packet->size=0;
    return sent<0 ? ERR_API_FAIL : receive(d,buffer,info);
}
/* One buffered picture per call; the game calls it until no picture comes back. */
static ABI int32_t videodec_flush(Ctrl *ctrl, FrameBuffer *buffer, PictureInfo *info) {
    int32_t r=check(ctrl,buffer,info);
    if (r) return r;
    Decoder *d=ctrl->handle;
    avcodec_send_packet(d->codec,NULL); /* AVERROR_EOF once draining: the rest still comes out */
    return receive(d,buffer,info);
}
static ABI int32_t videodec_reset(Ctrl *ctrl) {
    if (!ctrl) return ERR_ARGUMENT_POINTER;
    if (!ctrl->handle) return ERR_HANDLE;
    avcodec_flush_buffers(((Decoder *)ctrl->handle)->codec);
    return 0;
}
static ABI int32_t videodec_delete(Ctrl *ctrl) {
    if (!ctrl) return ERR_ARGUMENT_POINTER;
    Decoder *d=ctrl->handle;
    if (!d) return ERR_HANDLE;
    avcodec_free_context(&d->codec); av_packet_free(&d->packet); av_frame_free(&d->frame); av_frame_free(&d->nv12);
    sws_freeContext(d->sws);
    free(d);
    ctrl->handle=NULL;
    return 0;
}

static const RuntimeExport exports[]={
    {"sceVideodecQueryResourceInfo",videodec_query}, {"sceVideodecCreateDecoder",videodec_create},
    {"sceVideodecDecode",videodec_decode}, {"sceVideodecFlush",videodec_flush},
    {"sceVideodecReset",videodec_reset}, {"sceVideodecDeleteDecoder",videodec_delete},
};
uintptr_t runtime_videodec_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
