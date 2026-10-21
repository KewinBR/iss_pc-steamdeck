// SPDX-License-Identifier: GPL-2.0-or-later
// tlg: frame analyzer. Creating the file named by TLG_CAPTURE_TRIGGER records the next full
// frame: every render pass (targets, depth, draw count, shaders, sampled textures) and compute
// dispatch, in order, plus the buffer that was presented. Input for placing a temporal
// upscaler (scene color, depth, motion vectors, UI) in the game's frame.

#pragma once

#include <atomic>
#include <cstdio>
#include <string>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
struct ImageInfo;
}

namespace Vulkan {

class FrameCapture {
public:
    /// VideoOut thread, per flip: frame boundary, presented buffer, trigger file check.
    static void OnFlip(VAddr presented_address);

    /// VideoOut: a display buffer the game registered. A pass drawing into one starts a frame
    /// in the GPU command stream (the flip itself runs a frame behind).
    static void AddDisplayBuffer(VAddr address);
    static bool IsDisplayBuffer(VAddr address);

    /// GPU thread: whether passes are being looked at (armed or recording; cheap).
    static bool Active() {
        return state.load(std::memory_order_relaxed) != Idle;
    }
    /// GPU thread: whether a frame is being recorded right now.
    static bool IsRecording() {
        return state.load(std::memory_order_relaxed) == Recording;
    }
    /// GPU thread, at every draw/dispatch: starts or ends a recording at frame boundaries.
    static void Poll();

    /// `state`: the pass's census line (vk_pass_census.h), kept with the first pipeline that
    /// opened the pass.
    static void BeginPass(const VideoCore::ImageInfo* const* colors, u32 num_colors,
                          const VideoCore::ImageInfo* depth, std::string census);
    static void Draw(u64 vs_hash, u64 ps_hash, u32 num_indices, u32 num_instances);
    static void Dispatch(u64 cs_hash, u32 x, u32 y, u32 z);
    static void Sampled(const VideoCore::ImageInfo& info, bool storage);
    /// Contents of a bound buffer (first 1 KiB), kept for small passes and a pass's first draw.
    static void Buffer(u64 stage_hash, u32 slot, VAddr address, const void* data, u64 size);
    static void Note(const char* text);

    /// In-game menu: records the next full frame (as the trigger file does).
    static void Request();
    /// What the last request did, for the menu: empty, "Capturing...", the file or an error.
    static std::string LastResult();

private:
    enum : u32 { Idle, Armed, Recording };
    static inline std::atomic<u32> state{Idle};
    static inline std::atomic<u64> flips{0};
    static inline std::atomic<VAddr> last_presented{0};
    static inline std::atomic<bool> requested{false};
};

/// steam-deck: modo inspeção, identifica o objeto no centro da tela. Criar o arquivo TLG_INSPECT_TRIGGER (o
/// rodar-iss.sh usa /tmp/iss-inspecionar; L3+R3 juntos no controle criam ele) grava o próximo quadro como a captura
/// acima e põe em cada desenho uma consulta de oclusão limitada ao pixel do centro do alvo. O último desenho com cor
/// e profundidade que escreveu ali é o objeto mirado: o código dele (M-xxxx, dos índices e shaders) aparece na tela e
/// vai pro log e pra <captures>/inspecao_<data>.txt, com tudo o que passou pelo centro (endereço dos índices,
/// contagem, shaders, passe da captura do mesmo quadro). Só desenhos diretos (os indiretos não entram).
class Inspect {
public:
    struct DrawInfo {
        u64 vs{}, ps{};
        VAddr indices{}; ///< endereço do primeiro índice (0 se não indexado)
        u32 num_indices{}, instances{}, first_vertex{}, first_instance{};
        bool depth_write{}, color{};
        VAddr color0{};
        u16 width{}, height{};
    };
    /// GPU thread: se o quadro gravado é de inspeção.
    static bool Active() {
        return active.load(std::memory_order_relaxed);
    }
    /// GPU thread: registra o desenho; índice da consulta ou ~0u (sem espaço).
    static u32 Begin(vk::Device device, const DrawInfo& info);
    static vk::QueryPool Pool() {
        return pool;
    }

private:
    friend class FrameCapture;
    static inline std::atomic<bool> active{false};
    static inline vk::QueryPool pool{};
};

} // namespace Vulkan
