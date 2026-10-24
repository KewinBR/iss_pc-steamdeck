#ifdef TLG_RENDERER_V2
// Renderer v2 build has no perf trace (gpu/shim/tlg_perf_trace.cpp).
namespace TlgPerf {
inline bool Enabled() { return false; }
inline bool Mark(unsigned) { return false; }
} // namespace TlgPerf
#else
#include "tlg_perf_trace.h"
#endif
// SPDX-License-Identifier: GPL-2.0-or-later
#include "tlg_overlay.h"

#include <algorithm>
#include <string>
#include <filesystem>
#include <deque>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <vector>

#include <SDL3/SDL.h>
#include "tlg_settings.h"
#include "tlg_orbs.h"
#include "tlg_actors.h"
#include "tlg_graphics.h"
#include "tlg_perf_stats.h"
#include "../../src/actor_hook.h"

extern TlgActorSlot tlg_actor_slots[2];
extern "C" void runtime_restart(void); // src/probe.c: starts run.py again, ends this process
extern "C" void runtime_pause_guest(int pause); // src/runtime_thread.c (steam-deck)
extern "C" void tlggpu_set_guest_paused(int paused);
// steam-deck: View opens the menu (runtime_pad.c, the game paused); R3+L2 only with TLG_VIEW_MENU=0.
static bool ViewMenu() {
    static const bool on = [] {
        const char* e = std::getenv("TLG_VIEW_MENU");
        return !(e && e[0] == '0');
    }();
    return on;
}
#include "common/elf_info.h"
#ifndef TLG_RENDERER_V2
#include "common/backports.h"
#include "shader_recompiler/recompiler_toggles.h"
#endif
#include "imgui.h"
#include "imgui_internal.h" // RegisterUserTexture (save pictures)
#include "imgui_impl_vulkan.h"
#ifdef TLG_RENDERER_V2
#include "video_core/renderdoc.h" // RequestScreenshot (caderno das sessoes)
#endif
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#ifndef TLG_RENDERER_V2
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#endif

// DejaVu Sans (Cyrillic), embedded (third_party/fonts, Bitstream Vera license).
#ifdef _WIN32
asm(".section .rdata,\"dr\"\n"
    ".balign 16\n"
    ".global tlg_font_ttf\n"
    "tlg_font_ttf:\n"
    ".incbin \"" TLG_FONT_PATH "\"\n"
    ".global tlg_font_ttf_end\n"
    "tlg_font_ttf_end:\n"
    ".text\n");
#else
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden tlg_font_ttf\n"
    ".global tlg_font_ttf\n"
    "tlg_font_ttf:\n"
    ".incbin \"" TLG_FONT_PATH "\"\n"
    ".hidden tlg_font_ttf_end\n"
    ".global tlg_font_ttf_end\n"
    "tlg_font_ttf_end:\n"
    ".previous\n");
#endif
extern "C" const unsigned char tlg_font_ttf[];
extern "C" const unsigned char tlg_font_ttf_end[];

namespace TlgOverlay {
namespace {
// This overlay's own ImGui context: renderer v2's presenter composes the game image with ImGui
// in its own context, so each entry point here makes this one current and restores the other.
ImGuiContext* overlay_context = nullptr;
#ifdef TLG_RENDERER_V2
struct ContextScope {
    ImGuiContext* previous = ImGui::GetCurrentContext();
    ContextScope() {
        if (overlay_context) ImGui::SetCurrentContext(overlay_context);
    }
    ~ContextScope() {
        ImGui::SetCurrentContext(previous);
    }
};
#else
struct ContextScope {}; // this port's renderer: the overlay's context is the only one
#endif
} // namespace

namespace {

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
std::atomic<bool> menu_open{false};
std::atomic<int> open_request{-1}; // steam-deck: RequestOpen (pad thread) -> next present
bool r3_down = false, l2_down = false;
std::atomic<bool> loading{false};
std::atomic<u32> loading_done{0}, loading_total{0};
std::atomic<bool> loading_compile{false}; // first-run shader compilation (SetLoadingCompile)
std::chrono::steady_clock::time_point loading_compile_started{};
std::atomic<u32> background_done{0}, background_total{0};
bool dirty = false; // settings changed while open: saved on close
int requested_tab = -1; // switch on the next frame (L1/R1, Q/E, menu opened)
// The game's text dialog (SetTextEntry), guarded by imgui_mutex.
bool text_entry_active = false;
std::string text_entry_prompt, text_entry_text;
// The game's save dialogs (sceSaveDataDialog): the list, a message the player answers
// (overwrite?, no space...), the "Saving..." panel and the "Game saved" toast. The save's text
// comes from the game; every other word is the port's, in the menu language. Guarded by
// imgui_mutex; the game's thread polls the atomics.
using SaveClock = std::chrono::steady_clock;
std::atomic<bool> choice_active{false};  // the list
std::atomic<int> choice_result{-1};      // -1 open, -2 cancelled, else the chosen index
std::atomic<bool> message_active{false};
std::atomic<int> message_result{-1};     // -1 open, 1 yes/OK, 2 no/cancel
int save_kind = 1;                       // 1 save, 2 load, 3 delete
std::vector<SaveRow> save_rows;
int choice_index = 0, choice_first = 0;  // selected row, first row shown (the list scrolls)
int message_code = 0, message_buttons = 0, message_focus = 1; // focus: 0 no/cancel, 1 yes/OK
u64 message_value = 0;
std::string message_text;
bool message_has_entry = false;
SaveRow message_entry;
bool progress_on = false;
int progress_kind = 1, progress_percent = -1;
std::string progress_text;
SaveClock::time_point progress_since{}, progress_until{}; // until: a panel shown for a moment
SaveClock::time_point toast_since{}, toast_until{};
int toast_kind = 0;
// The mouse (display pixels) and the last frame's rectangles: the rows, the message's buttons.
ImVec2 save_mouse{-1.0f, -1.0f};
std::vector<ImRect> save_row_rects;
ImRect save_button_rects[2];
// Save pictures: decoded on the game's thread, made textures on the present thread (null =
// none). Closed dialogs' textures wait in choice_retired until the backend has destroyed them.
std::vector<TlgPng::Image> choice_pictures;
std::vector<ImTextureData*> choice_textures, choice_retired;
std::atomic<bool> choice_retiring{false};
float base_scale = 1.0f;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    ImGui::GetIO().MouseDrawCursor = value;
    if (value) {
        requested_tab = TlgSettings::Get().menu_tab;
    }
    if (!value && dirty) {
        dirty = false;
        TlgSettings::Save();
    }
}

ImGuiKey KeyFromSdl(SDL_Keycode key) {
    switch (key) {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_Q: return ImGuiKey_Q;
    case SDLK_E: return ImGuiKey_E;
    case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_RCTRL: return ImGuiKey_RightCtrl;
    case SDLK_LSHIFT: return ImGuiKey_LeftShift;
    case SDLK_RSHIFT: return ImGuiKey_RightShift;
    case SDLK_LALT: return ImGuiKey_LeftAlt;
    case SDLK_RALT: return ImGuiKey_RightAlt;
    default: return ImGuiKey_None;
    }
}

ImGuiKey KeyFromGamepad(u8 button) {
    switch (button) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    case SDL_GAMEPAD_BUTTON_BACK: return ImGuiKey_GamepadBack;
    default: return ImGuiKey_None;
    }
}

float PixelDensity(SDL_WindowID id) {
    SDL_Window* window = SDL_GetWindowFromID(id);
    const float density = window ? SDL_GetWindowPixelDensity(window) : 1.0f;
    return density > 0.0f ? density : 1.0f;
}

// Marks the settings dirty when a widget changed them.
template <typename T>
void Store(std::atomic<T>& target, T value, bool changed) {
    if (changed) {
        target = value;
        dirty = true;
    }
}

void Checkbox(const char* label, std::atomic<bool>& value) {
    bool v = value;
    const bool changed = ImGui::Checkbox(label, &v);
    Store(value, v, changed);
}

// A combo over `count` labels storing an index or a value from `values`.
void Choice(const char* label, std::atomic<int>& target, const char* const* labels, const int* values,
            int count) {
    int current = 0;
    for (int i = 0; i < count; ++i) {
        if ((values ? values[i] : i) == target.load()) current = i;
    }
    if (ImGui::Combo(label, &current, labels, count)) {
        Store(target, values ? values[current] : current, true);
    }
}

// Controller button names in the help text follow TLG_GLYPHS=xbox|nintendo (upstream 9155cdc;
// a port's runner can pick it from `tlg-probe --gamepad`); unset is PlayStation. Read once: the
// environment is fixed for a run.
static const char* PadText(const char* playstation, const char* xbox, const char* nintendo) {
    static const char* style = std::getenv("TLG_GLYPHS");
    if (!style) return playstation;
    return !std::strcmp(style, "xbox") ? xbox : !std::strcmp(style, "nintendo") ? nintendo : playstation;
}

// The note for the item under the mouse or the gamepad/keyboard focus, shown in the footer.
const char* footer_help = nullptr;

void Help(const char* text) {
    if (text && (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) || ImGui::IsItemFocused())) {
        footer_help = text;
    }
}

// Two-column rows: the name on the left, the control on the right.
bool BeginRows(const char* id) {
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingStretchProp)) {
        return false;
    }
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 0.52f);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.48f);
    return true;
}

void Row(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::PushID(label);
}

void CheckRow(const char* label, std::atomic<bool>& value, const char* help = nullptr) {
    Row(label);
    Checkbox("##value", value);
    Help(help);
    ImGui::PopID();
}

void ChoiceRow(const char* label, std::atomic<int>& target, const char* const* labels, const int* values,
               int count, const char* help = nullptr) {
    Row(label);
    Choice("##value", target, labels, values, count);
    Help(help);
    ImGui::PopID();
}

// A 0.1x to 100x multiplier with a 1x button.
void MultiplierRow(const char* label, std::atomic<float>& value, const char* help) {
    Row(label);
    const ImGuiStyle& style = ImGui::GetStyle();
    ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("1x").x + style.FramePadding.x * 2 + style.ItemSpacing.x));
    float v = value;
    if (ImGui::SliderFloat("##value", &v, 0.1f, 100.0f, "%.2fx",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        value = TlgOrbs::ClampMultiplier(v);
        dirty = true;
    }
    Help(help);
    ImGui::SameLine();
    if (ImGui::SmallButton("1x")) {
        value = 1.0f;
        dirty = true;
    }
    Help(help);
    ImGui::PopID();
}

void FloatRow(const char* label, std::atomic<float>& value, float lo, float hi, const char* format, const char* help,
              ImGuiSliderFlags flags = 0) {
    Row(label);
    float v = value;
    if (ImGui::SliderFloat("##value", &v, lo, hi, format, ImGuiSliderFlags_AlwaysClamp | flags)) {
        Store(value, std::clamp(v, lo, hi), true);
    }
    Help(help);
    ImGui::PopID();
}

void InfoRow(const char* label, const char* text) {
    Row(label);
    ImGui::TextDisabled("%s", text);
    ImGui::PopID();
}

void SliderRow(const char* label, std::atomic<int>& value, int lo, int hi, const char* format, const char* help) {
    Row(label);
    int v = value;
    if (ImGui::SliderInt("##value", &v, lo, hi, format, ImGuiSliderFlags_AlwaysClamp)) {
        Store(value, std::clamp(v, lo, hi), true);
    }
    Help(help);
    ImGui::PopID();
}

const char* UpscalerLabel(int upscaler) {
    switch (upscaler) {
    case TlgSettings::UpscalerFsr3: return "FSR 3.1";
    case TlgSettings::UpscalerFsr4: return "FSR 4";
    case TlgSettings::UpscalerFsr411: return "FSR 4.1.1";
    case TlgSettings::UpscalerTaa: return "TAA";
    case TlgSettings::UpscalerDlss: return "DLSS";
    default: return "";
    }
}


// tlg: menu language. TLG_UI_LANG=pt_BR or en_US; otherwise the system locale (LANG/LC_*):
// Portuguese when it starts with "pt", English otherwise.
bool UiPortuguese() {
    static const bool pt = [] {
        for (const char* name : {"TLG_UI_LANG", "LC_MESSAGES", "LANG"}) {
            if (const char* v = std::getenv(name); v && *v) return v[0] == 'p' && v[1] == 't';
        }
        return false;
    }();
    return pt;
}
const char* TL(const char* en, const char* pt) { return UiPortuguese() ? pt : en; }
const Vulkan::Instance* overlay_instance = nullptr; // for the overlay's measurements

// The performance overlay's text with the current settings (also the menu's preview).
std::string OverlayText() {
    namespace P = TlgPerfStats;
    const auto& s = TlgSettings::Get();
    const unsigned items = s.overlay_items;
    if ((items & (P::Gpu | P::Cpu | P::Ram | P::Vram)) && overlay_instance) {
        P::Start(*overlay_instance);
    }
    P::Sample sample = P::Latest();
    sample.fps = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
    sample.frame_ms = frame_ms_avg;
    sample.upscaler = UpscalerLabel(s.upscaler);
    return P::Format(sample, items, s.overlay_layout == 0);
}

constexpr const char* Unsupported = "Unavailable: game version or code signature mismatch.";


// steam-deck: the simple part of the Graphics tab. Quality presets and switches for the game patches
// (resolution, draw distance, shadows, particles, motion blur, reflections, the 30 FPS cap); the Deck's
// launch script reads them from tlg.ini and builds the patches on the next start.
void QualitySection() {
    auto& s = TlgSettings::Get();
    const bool pt = UiPortuguese();
    static const char* const presets_pt[] = {"Alto", "Médio", "Baixo", "Personalizado"};
    static const char* const presets_en[] = {"High", "Medium", "Low", "Custom"};
    static const char* const levels_pt[] = {"Alto", "Médio", "Baixo"};
    static const char* const levels_en[] = {"High", "Medium", "Low"};
    static const char* const res_pt[] = {"1920x1200", "1600x1000", "1280x800", "960x600"};
    static const char* const res_en[] = {"1920x1200", "1600x1000", "1280x800", "960x600"};
    static const char* const fovs[] = {"100%", "95%", "90%", "85%", "80%"};
    ImGui::SeparatorText(TL("Quality (applies when the game opens again)", "Qualidade (vale na próxima abertura do jogo)"));
    if (!BeginRows("##quality")) {
        return;
    }
    const int preset_before = s.gfx_preset;
    ChoiceRow(TL("Preset", "Predefinição"), s.gfx_preset, pt ? presets_pt : presets_en, nullptr, 4,
              TL("Sets everything below at once. High: the game as it is; Low: the most FPS.",
                 "Ajusta tudo abaixo de uma vez. Alto: o jogo como é; Baixo: o máximo de FPS."));
    if (s.gfx_preset != preset_before && s.gfx_preset < 3) {
        const int p = s.gfx_preset;
        s.gfx_resolution = p == 0 ? 0 : 2; // Médio e Baixo: o tamanho da tela do Deck
        s.gfx_distance = p;
        s.gfx_shadows = p;
        s.particles = p;
        s.gfx_motion_blur = p == 0;
        s.gfx_ssr = p < 2;
        dirty = true;
    }
    const int r = s.gfx_resolution, d = s.gfx_distance, sh = s.gfx_shadows, pa = s.particles;
    const bool mb = s.gfx_motion_blur, ssr = s.gfx_ssr;
    ChoiceRow(TL("Resolution", "Resolução"), s.gfx_resolution, pt ? res_pt : res_en, nullptr, 4,
              TL("Size the game draws at. 1280x800 is the Deck's screen in 16:10, with no black bars; the others "
                 "are 16:9 (bars above and below).",
                 "Tamanho em que o jogo desenha. 1280x800 é a tela do Deck em 16:10, sem barras pretas; as outras "
                 "são 16:9 (barras em cima e embaixo)."));
    ChoiceRow(TL("Draw distance", "Distância de visão"), s.gfx_distance, pt ? levels_pt : levels_en, nullptr, 3,
              TL("How far buildings and details appear. Lower draws less: more FPS.",
                 "Até onde aparecem prédios e detalhes. Menor desenha menos: mais FPS."));
    ChoiceRow(TL("Shadows", "Sombras"), s.gfx_shadows, pt ? levels_pt : levels_en, nullptr, 3,
              TL("Sun shadow detail and reach. Shadows are about half of what the game draws.",
                 "Detalhe e alcance das sombras do sol. As sombras são quase metade do que o jogo desenha."));
    ChoiceRow(TL("Particles", "Partículas"), s.particles, pt ? levels_pt : levels_en, nullptr, 3,
              TL("Fewer leftover effects and fewer particles per effect: powers and the special run faster.",
                 "Menos efeitos que sobram e menos partículas por efeito: poderes e o especial ficam mais leves."));
    CheckRow(TL("Motion blur", "Motion blur"), s.gfx_motion_blur,
             TL("Blur when the camera moves fast.", "Desfoque quando a câmera mexe rápido."));
    CheckRow(TL("Real-time reflections", "Reflexos em tempo real"), s.gfx_ssr,
             TL("Reflections on wet streets and glass (screen-space). Off: more FPS.",
                "Reflexos nas ruas molhadas e vidros (SSR). Desligado: mais FPS."));
    if (s.gfx_preset < 3 && (s.gfx_resolution != r || s.gfx_distance != d || s.gfx_shadows != sh ||
                             s.particles != pa || s.gfx_motion_blur != mb || s.gfx_ssr != ssr)) {
        s.gfx_preset = 3; // a single item changed: custom
        dirty = true;
    }
    ChoiceRow(TL("Field of view", "Campo de visão (FOV)"), s.gfx_fov, fovs, nullptr, 5,
              TL("Smaller shows less around Delsin and makes everything bigger on the small screen.",
                 "Menor mostra menos em volta do Delsin e deixa tudo maior na tela pequena."));
    CheckRow(TL("Unlock 30 FPS", "Destravar 30 FPS"), s.gfx_unlock_fps,
             TL("The game caps itself at 30 FPS; unlocked it goes as fast as the Deck allows.",
                "O jogo se trava em 30 FPS; destravado vai até onde o Deck aguentar."));
    CheckRow(TL("Show FPS", "Mostrar FPS"), s.show_fps,
             TL("The performance overlay in a corner of the screen (applies now).",
                "O contador de desempenho num canto da tela (vale na hora)."));
    ImGui::EndTable();
}

void ImageTab() {
    namespace G = TlgGraphics;
    auto& s = TlgSettings::Get();
    QualitySection();
    // Everything that was here before stays, under "Advanced".
    if (!ImGui::CollapsingHeader(TL("Advanced", "Avançado"))) {
        return;
    }
    ImGui::SeparatorText(TL("Display", "Tela"));
    if (BeginRows("##display")) {
        static const char* const modes[] = {TL("Windowed", "Janela"), TL("Borderless window", "Janela sem borda"), TL("Fullscreen", "Tela cheia")};
        ChoiceRow(TL("Display mode", "Modo de exibição"), s.display_mode, G::DisplayModeCount == 3 ? modes : G::DisplayModeNames.data(),
                  nullptr, G::DisplayModeCount, TL("Windowed, borderless window or exclusive fullscreen.", "Janela, janela sem borda ou tela cheia exclusiva."));
        CheckRow("VSync", s.vsync, TL("Waits for the monitor: no tearing, a little more input delay.", "Espera o monitor: sem tearing, um pouco mais de atraso de entrada."));
        ImGui::EndTable();
    }
    const int iw = s.image_width, ih = s.image_height, ww = s.window_width, wh = s.window_height;
    const bool upscaling = iw < ww && ih < wh;
    const bool downscaling = iw > ww || ih > wh;
    const bool rcas_blocked = s.rcas && !s.rcas_applied;
    // tlg: only spatial filters on the final image. The temporal upscalers (FSR 3/4, TAA, DLSS) were
    // built for GoW3's pipeline and make TLG's image shake, so they are not offered here.
    ImGui::SeparatorText(TL("Scaling and sharpening", "Escala e nitidez"));
    if (BeginRows("##scaling")) {
        CheckRow(TL("Upscale with FSR 1", "Ampliar com FSR 1"), s.fsr1,
                 TL("Enlarges the game image when it is smaller than the window (spatial filter, no shimmer).", "Amplia a imagem do jogo quando ela é menor que a janela (filtro espacial, sem tremor)."));
        ImGui::BeginDisabled(rcas_blocked);
        CheckRow(TL("Sharpening (RCAS)", "Nitidez (RCAS)"), s.rcas,
                 rcas_blocked && downscaling ? TL("No effect: the game's image is larger than the window.", "Sem efeito: a imagem do jogo é maior que a janela.")
                 : rcas_blocked              ? TL("Needs FSR 1: the game's image is smaller than the window.", "Precisa do FSR 1: a imagem do jogo é menor que a janela.")
                                             : TL("Sharpens the final image.", "Deixa a imagem final mais nítida."));
        ImGui::EndDisabled();
        ImGui::BeginDisabled(!s.rcas || rcas_blocked);
        SliderRow(TL("Sharpening strength", "Intensidade da nitidez"), s.rcas_strength, 0, 100, "%d%%", TL("How strong the sharpening is.", "Quão forte é a nitidez."));
        ImGui::EndDisabled();
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Draw distance and detail", "Distância de visão e detalhe"));
    if (BeginRows("##view")) {
        ImGui::BeginDisabled(!s.view_supported);
        FloatRow(TL("LOD distance", "Distância do LOD"), s.lod_scale, 0.25f, 4.0f, "%.2fx",
                 s.view_supported ? TL("How far objects keep their detailed models (every LOD band). Above 1x costs CPU and GPU; "
                                       "the game still lowers it on its own in heavy scenes.",
                                       "Até onde os objetos mantêm os modelos detalhados (todas as faixas de LOD). Acima de 1x pesa na CPU e na GPU; "
                                       "o jogo ainda reduz sozinho em cenas pesadas.")
                                  : TL(Unsupported, "Indisponível: versão do jogo ou assinatura do código diferente."));
        FloatRow(TL("Regions' LOD_ADJUST", "LOD_ADJUST das regiões"), s.region_lod, 0.0f, 1.0f, "%.2f",
                 TL("How much the map regions' own LOD changes count: 1 as in the game, 0 ignored.",
                    "Quanto contam as mudanças de LOD das regiões do mapa: 1 como no jogo, 0 ignoradas."));
        FloatRow(TL("Regions' FADE_DISTANCE", "FADE_DISTANCE das regiões"), s.region_fade, 0.25f, 4.0f, "%.2fx",
                 TL("Distance over which a region's lighting, haze and LOD settings blend in (multiplier).",
                    "Distância em que a luz, a névoa e o LOD de uma região se misturam ao entrar nela (multiplicador)."));
        FloatRow(TL("Near clip (NEAR_CLIP)", "Plano próximo (NEAR_CLIP)"), s.near_clip, 1.0f, 100.0f, "%.1f",
                 TL("Default near plane of the cameras (the game uses 10). Lower shows objects closer to the camera, "
                    "higher reduces depth flicker far away. Applies to the next camera (load, cutscene).",
                    "Plano próximo padrão das câmeras (o jogo usa 10). Menor mostra objetos mais perto da câmera, "
                    "maior reduz o tremido de profundidade ao longe. Vale na próxima câmera (carregamento, cena)."));
        FloatRow(TL("Far clip (FAR_CLIP)", "Plano distante (FAR_CLIP)"), s.far_clip, 1000.0f, 1.0e7f, "%.0f",
                 TL("Default far plane of the cameras (the game uses 10000000, unlimited). Lower cuts the view; "
                    "applies to the next camera.",
                    "Plano distante padrão das câmeras (o jogo usa 10000000, sem limite). Menor corta a visão; "
                    "vale na próxima câmera."),
                 ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
        FloatRow(TL("Texture LOD bias (MipLodBias)", "Bias de LOD das texturas (MipLodBias)"), s.mip_lod_bias, -3.0f, 3.0f,
                 "%+.2f",
                 TL("Negative: sharper textures at a distance (more shimmer, a little more GPU). Positive: blurrier.",
                    "Negativo: texturas mais nítidas ao longe (mais cintilação, um pouco mais de GPU). Positivo: mais borradas."));
        Row(TL("Restore the game's values", "Voltar aos valores do jogo"));
        if (ImGui::SmallButton(TL("Restore", "Restaurar"))) {
            s.lod_scale = 1.0f;
            s.region_lod = 1.0f;
            s.region_fade = 1.0f;
            s.near_clip = 10.0f;
            s.far_clip = 1.0e7f;
            s.mip_lod_bias = 0.0f;
            const float bands[6] = {400.0f, 750.0f, 1500.0f, 3500.0f, 6000.0f, 15000.0f};
            for (int b = 0; b < 6; ++b) s.lod_bands[b] = bands[b];
            dirty = true;
        }
        ImGui::PopID();
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Information", "Informações"));
    if (BeginRows("##image_info")) {
        char text[96];
        std::snprintf(text, sizeof(text), "%d x %d", iw, ih);
        InfoRow(TL("Game image", "Imagem do jogo"), text);
        std::snprintf(text, sizeof(text), "%d x %d", ww, wh);
        InfoRow(TL("Windowed", "Janela"), text);
        InfoRow("FSR 1", !s.fsr1 ? TL("off", "desligado") : upscaling ? TL("active", "ativo") : TL("no effect (image not smaller than window)", "sem efeito (imagem não é menor que a janela)"));
        InfoRow(TL("Sharpening", "Nitidez"), s.rcas_applied ? TL("applied", "aplicada") : TL("not applied", "não aplicada"));
        ImGui::EndTable();
    }
}

void FixesSection();

void PerformanceTab() {
    if (TlgPerf::Enabled()) {
        ImGui::SeparatorText(TL("Performance capture", "Captura de desempenho"));
        static int category = 0;
        const char* labels[] = {TL("Absorb smoke", "Absorver fumaça"), TL("Use power", "Usar poder"),
                                TL("Movement", "Movimentação"), TL("Other", "Outro")};
        ImGui::Combo(TL("Event", "Evento"), &category, labels, 4);
        if (ImGui::Button(TL("Mark performance event", "Marcar evento de desempenho"))) {
            const bool accepted = TlgPerf::Mark(static_cast<unsigned>(category + 1));
            TlgOverlay::Notice(accepted ? TL("Event marked", "Evento marcado") :
                                          TL("Capture queue full", "Fila de captura cheia"), 2.0f);
        }
    }
    namespace G = TlgGraphics;
    auto& s = TlgSettings::Get();
    // tlg: GoW3's render-resolution and engine-frame-rate patches and the readback shortcuts are not
    // offered: they do not apply to TLG, and the readback ones crash it (it needs precise readbacks).
    ImGui::SeparatorText(TL("Live", "Em tempo real"));
    if (BeginRows("##live")) {
        static const char* const fps_labels[] = {"30", "120", "240", TL("Unlimited", "Sem limite")};
        ChoiceRow(TL("Frame rate limit", "Limite de FPS"), s.fps_limit, fps_labels, G::FpsLimits.data(), int(G::FpsLimits.size()),
                  TL("Caps the presented frames. The game runs at 30 FPS (as on PS4).", "Limita os quadros apresentados. O jogo roda a 30 FPS (como no PS4)."));
        static const char* const queued_labels[] = {TL("1 (lowest latency)", "1 (menor atraso)"), "2", TL("Unlimited", "Sem limite")};
        ChoiceRow(TL("Frames queued", "Quadros na fila"), s.frames_queued, queued_labels, G::FramesQueued.data(),
                  int(G::FramesQueued.size()), TL("More frames queued: steadier FPS, slower response to input.", "Mais quadros na fila: FPS mais estável, resposta mais lenta."));
        CheckRow(TL("Compile new shaders in the background", "Compilar shaders novos em segundo plano"), s.async_shaders,
                 TL("Fewer stutters in new areas; an object or effect may appear a moment later.", "Menos engasgos em áreas novas; um objeto ou efeito pode aparecer um instante depois."));
        ImGui::EndTable();
    }
    FixesSection();
}


// tlg: the fixes (common/backports.h) and the A/B switches. Live ones change at once; the others
// are only recorded in tlg.ini (bpNN=0/1, wqm=0/1, ...) and apply on the next start, when the
// launcher turns the keys into TLG_BPnn / TLG_<NAME>. The table is iterated, so a reordered
// backports.h still lists every entry.
#ifdef TLG_RENDERER_V2
// Renderer v2 has its own fixes and switches (ISS_* variables): none of this port's.
void FixesSection() {
    ImGui::SeparatorText(TL("Fixes", "Correções"));
    ImGui::TextDisabled("%s", TL("This build uses renderer v2: its own fixes apply.",
                                 "Esta versão usa o renderer v2: valem as correções dele."));
}
#else
void FixesSection() {
    namespace B = Common::Backports;
    auto& s = TlgSettings::Get();
    ImGui::SeparatorText(TL("Fixes", "Correções"));
    ImGui::TextDisabled("%s", TL("Marked (next start) are saved now and apply the next time the game starts.",
                                 "As marcadas (próxima partida) são gravadas agora e valem na próxima vez que o jogo abrir."));
    const ImVec4 amber(0.85f, 0.72f, 0.45f, 1.0f);
    if (ImGui::BeginTable("##fixes", 4, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch, 0.50f);
        ImGui::TableSetupColumn("on", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("hits", ImGuiTableColumnFlags_WidthStretch, 0.14f);
        ImGui::TableSetupColumn("source", ImGuiTableColumnFlags_WidthStretch, 0.36f);
        for (std::size_t i = 0; i < B::kEntries.size(); ++i) {
            const B::Entry& e = B::kEntries[i];
            const B::Id id = static_cast<B::Id>(i);
            const bool live = B::IsLive(id);
            const bool running = B::On(id);
            const int pref = i < std::size_t(TlgSettings::kBackportPrefs) ? s.backport_pref[i].load() : -1;
            bool shown = live || pref < 0 ? running : pref != 0;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%s  %s", e.id, e.what);
            ImGui::TableNextColumn();
            ImGui::PushID(e.id);
            if (ImGui::Checkbox("##on", &shown)) {
                if (live) {
                    B::SetLive(id, shown);
                }
                if (i < std::size_t(TlgSettings::kBackportPrefs)) {
                    s.backport_pref[i] = shown ? 1 : 0;
                    dirty = true;
                }
            }
            Help(live ? TL("Applies at once.", "Vale na hora.")
                      : TL("Applies on next start.", "Vale na próxima partida."));
            ImGui::PopID();
            if (!live) {
                ImGui::SameLine();
                const bool differs = shown != running;
                ImGui::TextColored(differs ? amber : ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "%s",
                                   TL("(next start)", "(próxima partida)"));
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%llu", static_cast<unsigned long long>(B::Hits(id)));
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", e.source);
        }
        ImGui::EndTable();
    }

    // A/B switches read once at start (recompiler or device creation): next start only.
    ImGui::SeparatorText(TL("A/B switches (next start)", "Chaves A/B (próxima partida)"));
    struct Extra { const char* name; bool (*current)(); };
    static const Extra extras[] = {
        {"WQM", Shader::Toggles::WholeQuadMode},
        {"LDS_SMALL", Shader::Toggles::SmallGroupLdsBarriers},
        {"LDS_MULTI", Shader::Toggles::MultiWaveLdsBarriers},
        {"LDS_READLANE", Shader::Toggles::LdsUniformReadLane},
        {"LANE_DIVERGENCE", Shader::Toggles::LaneDivergence},
        {"RFL_ALL", Shader::Toggles::ReadFirstLaneAll},
        {"BLOCK1D", nullptr},
        {"BC_TINY", nullptr},
        {"NO_MSAA", nullptr},
        {"DYNAMIC_STRIDE", Shader::Toggles::DynamicBufferStride},
    };
    static_assert(sizeof(extras) / sizeof(extras[0]) == TlgSettings::kExtraPrefs);
    if (BeginRows("##extras")) {
        static const char* const choices[] = {TL("Default", "Padrão"), TL("Off", "Desligado"), TL("On", "Ligado")};
        static const int values[] = {-1, 0, 1};
        for (int e = 0; e < TlgSettings::kExtraPrefs; ++e) {
            char label[64];
            std::snprintf(label, sizeof(label), "%s%s", extras[e].name,
                          extras[e].current ? (extras[e].current() ? TL("  (running: on)", "  (rodando: ligado)")
                                                                    : TL("  (running: off)", "  (rodando: desligado)"))
                                            : "");
            ChoiceRow(label, s.extra_pref[e], choices, values, 3,
                      TL("Read once at start; saved in tlg.ini and applied on the next start.",
                         "Lida uma vez ao iniciar; gravada no tlg.ini e aplicada na próxima partida."));
        }
        ImGui::EndTable();
    }
}
#endif

void OpenGuide(); // the graffiti guide (below, with the save dialogs)

// steam-deck: recursos que nao fazem parte do jogo original; os experimentais vem desligados.
void ExtrasTab() {
    auto& s = TlgSettings::Get();
    ImGui::SeparatorText("Experimental");
    if (BeginRows("##extras")) {
        CheckRow(TL("Skip videos (Experimental)", "Pular vídeos (Experimental)"), s.movie_skip,
                 TL("During a cutscene video, hold the menu button for 1 s to skip it; a quick tap still pauses. "
                    "The rest of the scene's audio and subtitles may still play for a few seconds.",
                    "Durante um vídeo de cena, segure o botão de menu por 1 s para pular; um toque rápido continua "
                    "pausando. O resto do áudio e das legendas da cena pode tocar por alguns segundos."));
        ImGui::EndTable();
    }
    ImGui::SeparatorText("Easter Eggs");
    if (BeginRows("##easter")) {
        CheckRow("Easter Eggs", s.easter_eggs,
                 TL("Some graffiti of the spray minigame get a surprise. Applies to the graffiti loaded after the "
                    "change; restart the game to be sure.",
                    "Alguns grafites do minigame de spray ganham uma surpresa. Vale para os grafites carregados "
                    "depois da mudança; reinicie o jogo para garantir."));
        ImGui::EndTable();
    }
}

void ControlsTab() {
    auto& s = TlgSettings::Get();
    // Mouse (src/runtime_pad.c): the camera settings apply at once. The buttons and the touchpad
    // mode are fixed unless TLG_KEY_MAP / TLG_MOUSE_PAD_* (or tlg.ini key_map) change them.
    ImGui::SeparatorText(TL("Mouse", "Mouse"));
    if (BeginRows("##mouse")) {
        CheckRow(TL("Mouse turns the camera", "Mouse gira a câmera"), s.mouse_enabled,
                 TL("The game window takes the mouse while it has focus. Esc releases it; a click in the window takes it back.",
                    "A janela do jogo prende o mouse enquanto tem o foco. Esc solta; um clique na janela prende de novo."));
        ImGui::BeginDisabled(!s.mouse_enabled);
        FloatRow(TL("Sensitivity", "Sensibilidade"), s.mouse_sensitivity, 0.1f, 5.0f, "%.2f",
                 TL("How far the camera turns for the same mouse movement.", "Quanto a câmera gira para o mesmo movimento do mouse."));
        CheckRow(TL("Invert Y axis", "Inverter eixo Y"), s.mouse_invert_y,
                 TL("Moving the mouse forward looks down.", "Empurrar o mouse para frente olha para baixo."));
        FloatRow(TL("Smoothing", "Suavização"), s.mouse_smoothing, 0.0f, 0.9f, "%.2f",
                 TL("Evens out the camera motion; higher values feel softer and a little slower.",
                    "Deixa o movimento da câmera mais uniforme; valores altos ficam mais suaves e um pouco mais lentos."));
        FloatRow(TL("Acceleration", "Aceleração"), s.mouse_accel, 0.0f, 2.0f, "%.2f",
                 TL("Fast movements turn more than slow ones. 0 is linear.", "Movimentos rápidos giram mais que os lentos. 0 é linear."));
        FloatRow(TL("Anti-dead zone", "Anti-zona morta"), s.mouse_deadzone, 0.0f, 0.5f, "%.2f",
                 TL("Small mouse movements skip the game's stick dead zone. Raise it if slow movements do nothing.",
                    "Movimentos pequenos do mouse pulam a zona morta do analógico do jogo. Aumente se movimentos lentos não fizerem nada."));
        ImGui::EndDisabled();
        InfoRow(TL("Left / right / middle", "Esquerdo / direito / do meio"), "R2 / L2 / R3");
        InfoRow(TL("Touchpad mode", "Modo touchpad"), TL("hold Left Ctrl or Mouse X1", "segure Ctrl esquerdo ou botão X1 do mouse"));
        InfoRow(TL("In touchpad mode", "No modo touchpad"),
                TL("left touch/click/drag, middle touch, right two fingers", "esquerdo toca/clica/arrasta, do meio toca, direito dois dedos"));
        InfoRow(TL("Release the mouse", "Soltar o mouse"), "Esc");
        const char* env = std::getenv("TLG_KEY_MAP");
        const std::string remap = env && env[0] ? std::string(env) : s.key_map;
        if (!remap.empty()) {
            InfoRow(TL("Remapped", "Remapeado"), remap.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Sensor mode (graffiti)", "Modo sensor (grafite)"));
    if (BeginRows("##sensor")) {
        SliderRow(TL("Graffiti aiming speed", "Velocidade da mira do grafite"), s.sensor_aim, 25, 200, "%d%%",
                  TL("How fast the stick moves the graffiti cursor in sensor mode (100% = default).",
                     "Quão rápido o analógico move o cursor do grafite no modo sensor (100% = padrão)."));
        ImGui::EndTable();
    }
    if (ImGui::Button(TL("How to do the graffiti without a gyro", "Como fazer o grafite sem giroscópio"))) {
        SetOpen(false);
        OpenGuide();
    }
    Help(TL("Shows the steps again (the same screen as the first time).",
            "Mostra os passos de novo (a mesma tela da primeira vez)."));
}

void OverlayTab() {
    namespace P = TlgPerfStats;
    auto& s = TlgSettings::Get();
    if (BeginRows("##overlay_on")) {
        CheckRow(TL("Show overlay (F10)", "Mostrar overlay (F10)"), s.show_fps, TL("The performance overlay in a corner of the screen.", "O overlay de desempenho num canto da tela."));
        ImGui::EndTable();
    }
    ImGui::BeginDisabled(!s.show_fps);
    ImGui::SeparatorText(TL("What to show", "O que mostrar"));
    if (ImGui::BeginTable("##overlay_items", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableNextColumn();
        static const char* const names[] = {"FPS", TL("Frametime", "Tempo de quadro"), "Upscaler", TL("GPU (name and use)", "GPU (nome e uso)"), TL("CPU (game)", "CPU (jogo)"),
                                            TL("RAM (game / total)", "RAM (jogo / total)"), TL("VRAM (used / available)", "VRAM (usada / disponível)")};
        for (unsigned item = 0; item < 7; ++item) {
            const unsigned bit = 1u << item;
            bool on = (s.overlay_items & bit) != 0;
            if (ImGui::Checkbox(names[item], &on)) {
                s.overlay_items = on ? (s.overlay_items | bit) : (s.overlay_items & ~bit);
                dirty = true;
            }
        }
        ImGui::TableNextColumn();
        ImGui::TextDisabled(TL("Preview", "Prévia"));
        ImGui::BeginChild("##preview", ImVec2(0, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
        ImGui::SetWindowFontScale(s.overlay_scale / 100.0f);
        const std::string text = OverlayText();
        ImGui::TextUnformatted(text.empty() ? TL("(nothing selected)", "(nada selecionado)") : text.c_str());
        ImGui::SetWindowFontScale(1.0f);
        ImGui::EndChild();
        ImGui::EndTable();
    }
    ImGui::SeparatorText(TL("Appearance", "Aparência"));
    if (BeginRows("##appearance")) {
        SliderRow(TL("Font size", "Tamanho da fonte"), s.overlay_scale, 50, 300, "%d%%", TL("Size of the overlay text.", "Tamanho do texto do overlay."));
        static const char* const corners[] = {TL("Top left", "Superior esquerdo"), TL("Top right", "Superior direito"), TL("Bottom left", "Inferior esquerdo"), TL("Bottom right", "Inferior direito")};
        ChoiceRow(TL("Corner", "Canto"), s.overlay_corner, corners, nullptr, 4, TL("Where the overlay sits.", "Onde o overlay fica."));
        SliderRow(TL("Background opacity", "Opacidade do fundo"), s.overlay_opacity, 0, 100, "%d%%", TL("0% is no background.", "0% é sem fundo."));
        static const char* const layouts[] = {TL("One line", "Uma linha"), TL("One item per line", "Um item por linha")};
        ChoiceRow(TL("Layout", "Disposição"), s.overlay_layout, layouts, nullptr, 2, TL("Items side by side or stacked.", "Itens lado a lado ou empilhados."));
        ImGui::EndTable();
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText(TL("About", "Sobre"));
    if (BeginRows("##about")) {
        const auto& elf = Common::ElfInfo::Instance();
        char text[160];
        std::snprintf(text, sizeof(text), "%.*s v%.*s", int(elf.GameSerial().size()), elf.GameSerial().data(),
                      int(elf.AppVer().size()), elf.AppVer().data());
        InfoRow(TL("Game", "Jogo"), text);
        const u32 total = background_total;
        if (total) {
            std::snprintf(text, sizeof(text), TL("loading %u%%", "carregando %u%%"), u32(u64(background_done) * 100 / total));
        } else {
            std::snprintf(text, sizeof(text), TL("ready", "pronto"));
        }
        InfoRow(TL("Shader cache", "Cache de shaders"), text);
        ImGui::EndTable();
    }
}

// Startup options changed since this launch (applied by "Apply and restart").
int RestartChanges() {
    const auto& s = TlgSettings::Get();
    return (s.render_resolution != s.startup_render_resolution) + (s.engine_fps != s.startup_engine_fps) +
           (s.deferred_readback != s.startup_deferred_readback) + (s.stale_readback != s.startup_stale_readback);
}

// ---- steam-deck: o menu do port no estilo do menu de pausa do inFAMOUS Second Son ----
// Tela cheia sobre o último quadro do jogo (congelado pelo View): lista "OPÇÕES" à esquerda, explicação da opção
// à direita, abas no topo (L1/R1) e dicas de botão embaixo. Desenhado direto na lista de desenho do ImGui (sem
// janelas nem navegação do ImGui). Animações: os painéis crescem a partir do lado oposto, topo desce, rodapé sobe,
// valor dá um "pop" ao mudar, interruptor desliza, aba ativa cresce e fica amarela, L1/R1 piscam em laranja.
// O menu antigo continua no item "Menu avançado". TLG_CLASSIC_MENU=1 abre sempre o antigo.
namespace DeckMenuImpl {

ImFont* font_regular = nullptr;
ImFont* font_bold = nullptr;
double opened_at = 0.0;
bool classic = false; // "Menu avançado" escolhido: o menu antigo até fechar
bool shown = false;   // o menu novo já foi desenhado desde que abriu (a entrada começa no primeiro quadro)
// Diálogo de fechar: 0 nenhum, 1 "salvar?", 2 "reiniciar agora?", 3 aviso "reinicie" com contagem.
int dialog = 0, dialog_sel = 0;
double closing_at = -1.0; // fechando: menu some, pausa, escurecimento some, depois o jogo volta
double dialog_at = 0.0;
std::atomic<bool> close_request{false}; // View com o menu aberto (thread da janela)
struct Snapshot {
    int preset, resolution, distance, shadows, particles, fov, subtitles;
    bool motion_blur, ssr, unlock, fps, movie, easter;
};
Snapshot snap{};
Snapshot Take() {
    const auto& s = TlgSettings::Get();
    return {s.gfx_preset, s.gfx_resolution, s.gfx_distance, s.gfx_shadows, s.particles, s.gfx_fov,
            s.gfx_subtitles, s.gfx_motion_blur, s.gfx_ssr, s.gfx_unlock_fps, s.show_fps, s.movie_skip,
            s.easter_eggs};
}
void Restore(const Snapshot& v) {
    auto& s = TlgSettings::Get();
    s.gfx_preset = v.preset; s.gfx_resolution = v.resolution; s.gfx_distance = v.distance;
    s.gfx_shadows = v.shadows; s.particles = v.particles; s.gfx_fov = v.fov; s.gfx_subtitles = v.subtitles;
    s.gfx_motion_blur = v.motion_blur; s.gfx_ssr = v.ssr; s.gfx_unlock_fps = v.unlock; s.show_fps = v.fps;
    s.movie_skip = v.movie; s.easter_eggs = v.easter;
}
// Mudou alguma coisa? (restart: só as que valem na próxima abertura; Mostrar FPS e os Extras valem na hora)
bool Changed(bool restart_only) {
    const Snapshot n = Take();
    const bool gfx = n.preset != snap.preset || n.resolution != snap.resolution || n.distance != snap.distance ||
                     n.shadows != snap.shadows || n.particles != snap.particles || n.fov != snap.fov ||
                     n.subtitles != snap.subtitles ||
                     n.motion_blur != snap.motion_blur || n.ssr != snap.ssr || n.unlock != snap.unlock;
    return gfx || (!restart_only && (n.fps != snap.fps || n.movie != snap.movie || n.easter != snap.easter));
}
bool CanRestart() {
    const char* c = std::getenv("TLG_RESTART_COMMAND");
    return c && *c;
}

enum class Kind { Section, Choice, Switch, Fixed, Action };
struct Effect {
    const char* label;
    const char* value;
    int tone; // 0 normal, 1 bom (verde), 2 destaque (amarelo)
};
struct Item {
    Kind kind;
    const char* label;
    const char* help = "";
    std::atomic<int>* choice = nullptr;
    const char* const* names = nullptr;
    int count = 0;
    std::atomic<bool>* flag = nullptr;
    const char* fixed = "";
    std::vector<Effect> effects;
    int id = 0; // ações e regras especiais
};

constexpr u32 Branco = IM_COL32(233, 231, 225, 255);
constexpr u32 Texto = IM_COL32(217, 214, 207, 255);
constexpr u32 Apagado = IM_COL32(139, 138, 134, 255);
constexpr u32 Amarelo = IM_COL32(242, 182, 49, 255);
constexpr u32 Laranja = IM_COL32(240, 138, 36, 255);
constexpr u32 Verde = IM_COL32(159, 211, 107, 255);
constexpr u32 Painel = IM_COL32(14, 15, 18, 200);
constexpr u32 Faixa = IM_COL32(0, 0, 0, 184);

int tab = 0;
int sel = -1;
float scroll = 0.0f, scroll_target = 0.0f;
double tab_changed_at = -10.0, l1_at = -10.0, r1_at = -10.0, sel_changed_at = -10.0;
int prev_tab = 0;
std::array<std::array<double, 32>, 4> value_changed{}; // pop do valor, por aba e item
std::array<std::array<double, 32>, 4> switch_changed{};

float Ease(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}
u32 Alpha(u32 color, float a) {
    const u32 alpha = u32(std::clamp(a, 0.0f, 1.0f) * float((color >> IM_COL32_A_SHIFT) & 0xFF));
    return (color & ~IM_COL32_A_MASK) | (alpha << IM_COL32_A_SHIFT);
}
u32 Mix(u32 a, u32 b, float t) {
    const auto ch = [&](int s) {
        const float x = float((a >> s) & 0xFF), y = float((b >> s) & 0xFF);
        return u32(x + (y - x) * std::clamp(t, 0.0f, 1.0f)) << s;
    };
    return ch(0) | ch(8) | ch(16) | ch(24);
}

std::vector<Item> Build(int which) {
    auto& s = TlgSettings::Get();
    const bool pt = UiPortuguese();
    static const char* const levels_pt[] = {"Alto", "Médio", "Baixo"};
    static const char* const levels_en[] = {"High", "Medium", "Low"};
    static const char* const presets_pt[] = {"Alto", "Médio", "Baixo", "Personalizado"};
    static const char* const presets_en[] = {"High", "Medium", "Low", "Custom"};
    static const char* const res[] = {"1920x1200", "1600x1000", "1280x800", "960x600"};
    static const char* const fovs[] = {"100%", "95%", "90%", "85%", "80%"};
    const char* const* levels = pt ? levels_pt : levels_en;
    std::vector<Item> v;
    if (which == 0) {
        v.push_back({Kind::Section, TL("Quality", "Qualidade")});
        v.push_back({Kind::Choice, TL("Preset", "Predefinição"),
                     TL("Sets everything below at once. High is the game as made for the PS4; Low is the most FPS on the Deck.",
                        "Ajusta tudo abaixo de uma vez. Alto é o jogo como foi feito pro PS4; Baixo é o máximo de FPS no Deck."),
                     &s.gfx_preset, pt ? presets_pt : presets_en, 4, nullptr, "",
                     {{TL("Applies", "Vale"), TL("when the game opens again", "na próxima abertura"), 0}}, 1});
        v.push_back({Kind::Choice, TL("Resolution", "Resolução"),
                     TL("Size the game draws at, all in the Deck's 16:10 shape: no black bars, no stretching. 1280x800 is the screen's own size; higher is sharper and heavier, 960x600 is the lightest.",
                        "Tamanho em que o jogo desenha, todos no formato 16:10 do Deck: sem barras pretas e sem esticar. 1280x800 é o tamanho da tela; acima fica mais nítido e mais pesado, 960x600 é o mais leve."),
                     &s.gfx_resolution, res, 4, nullptr, "",
                     {{TL("Picture", "Imagem"), TL("always full screen (16:10)", "sempre tela cheia (16:10)"), 0},
                      {TL("Deck screen", "Tela do Deck"), "1280x800", 0}}, 2});
        v.push_back({Kind::Choice, TL("Draw distance", "Distância de visão"),
                     TL("How far buildings and details appear. On Low the horizon cards come in sooner; large pieces close by stay whole.",
                        "Até onde aparecem prédios e detalhes. No Baixo, os cartões do horizonte entram mais cedo; peças grandes perto continuam inteiras."),
                     &s.gfx_distance, levels, 3, nullptr, "",
                     {{TL("Draws", "Desenhos"), TL("-30% to -40% on Low", "-30% a -40% no Baixo"), 1}}, 2});
        v.push_back({Kind::Choice, TL("Shadows", "Sombras"),
                     TL("Sun shadow detail and reach. Shadows are about a third of what the game draws each frame.",
                        "Detalhe e alcance das sombras do sol. As sombras são um terço do que o jogo desenha em cada quadro."),
                     &s.gfx_shadows, levels, 3, nullptr, "", {}, 2});
        v.push_back({Kind::Choice, TL("Particles", "Partículas"),
                     TL("Fewer leftover effects and fewer particles per effect: powers and the special run much lighter.",
                        "Menos efeitos que sobram e menos partículas por efeito: poderes e o especial ficam bem mais leves."),
                     &s.particles, levels, 3, nullptr, "",
                     {{TL("Neon special", "Especial de neon"), TL("much lighter", "bem mais leve"), 1}}, 2});
        v.push_back({Kind::Choice, TL("Field of view", "Campo de visão"),
                     TL("Smaller shows less around Delsin and makes everything bigger on the Deck's small screen.",
                        "Menor mostra menos em volta do Delsin e deixa tudo maior na tela pequena do Deck."),
                     &s.gfx_fov, fovs, 5});
        static const char* const subs_en[] = {"Original", "Medium", "Large", "Huge"};
        static const char* const subs_pt[] = {"Original", "Média", "Grande", "Enorme"};
        v.push_back({Kind::Choice, TL("Subtitle size", "Tamanho da legenda"),
                     TL("The game's subtitles were made for a TV; on the Deck's screen Large reads easily. Long lines wrap a little wider.",
                        "As legendas do jogo foram feitas pra TV; na tela do Deck o Grande lê fácil. Linhas longas quebram um pouco mais largas."),
                     &s.gfx_subtitles, pt ? subs_pt : subs_en, 4});
        v.push_back({Kind::Section, TL("Effects", "Efeitos")});
        v.push_back({Kind::Switch, "Motion blur",
                     TL("Blur when the camera moves fast. Off looks sharper on a small screen.",
                        "Desfoque quando a câmera mexe rápido. Desligado fica mais nítido em tela pequena."),
                     nullptr, nullptr, 0, &s.gfx_motion_blur, "", {}, 2});
        v.push_back({Kind::Switch, TL("Real-time reflections", "Reflexos em tempo real"),
                     TL("Reflections on wet streets and glass. Off gains FPS in the rain.",
                        "Reflexos nas ruas molhadas e nos vidros. Desligado ganha FPS na chuva."),
                     nullptr, nullptr, 0, &s.gfx_ssr, "", {{TL("In the rain", "Na chuva"), "+FPS", 1}}, 2});
        v.push_back({Kind::Switch, TL("Unlock 30 FPS", "Destravar 30 FPS"),
                     TL("The game caps itself at 30 FPS; unlocked it goes as fast as the Deck allows.",
                        "O jogo se trava em 30 FPS. Destravado vai até onde o Deck aguentar."),
                     nullptr, nullptr, 0, &s.gfx_unlock_fps});
        v.push_back({Kind::Switch, TL("Show FPS", "Mostrar FPS"),
                     TL("The performance counter in a corner of the screen. Applies right away.",
                        "O contador de desempenho num canto da tela. Vale na hora."),
                     nullptr, nullptr, 0, &s.show_fps});
        v.push_back({Kind::Section, TL("Advanced", "Avançado")});
        v.push_back({Kind::Action, TL("Advanced menu", "Menu avançado"),
                     TL("Every technical option of the port (overlay, upscaler, performance switches, controls).",
                        "Todas as opções técnicas do port (overlay, upscaler, chaves de desempenho, controles)."),
                     nullptr, nullptr, 0, nullptr, TL("Open", "Abrir"), {}, 10});
    } else if (which == 1) {
        v.push_back({Kind::Section, TL("Graffiti and sensors", "Grafite e sensores")});
        v.push_back({Kind::Fixed, TL("Shake the Deck", "Chacoalhar o Deck"),
                     TL("Shaking the Deck up and down fills the spray can. Shake firmly for a moment.",
                        "Sacudir o Deck pra cima e pra baixo enche a lata de spray. Sacuda firme por um tempinho."),
                     nullptr, nullptr, 0, nullptr, TL("On", "Ligado")});
        v.push_back({Kind::Fixed, TL("Tilt", "Inclinação"),
                     TL("The Deck's gyroscope turns and aims the can, like the PS4 controller.",
                        "O giroscópio do Deck vira e mira a lata, como o controle do PS4."),
                     nullptr, nullptr, 0, nullptr, TL("Gyroscope", "Giroscópio")});
        v.push_back({Kind::Section, "Trackpads"});
        v.push_back({Kind::Fixed, TL("Touch and click", "Toque e clique"),
                     TL("The two trackpads are the two halves of the PS4 touchpad. Pressing both counts as two fingers.",
                        "Os dois trackpads são as duas metades do touchpad do PS4. Apertar os dois juntos vale como dois dedos."),
                     nullptr, nullptr, 0, nullptr, TL("PS4 touchpad", "Touchpad do PS4")});
        v.push_back({Kind::Section, TL("Back buttons", "Botões traseiros")});
        static const char* const back_help_pt = "Os botões de trás do Deck repetem os botões da frente, pra pular, atacar e usar poderes sem tirar o polegar do analógico. Os ícones do jogo continuam os da frente.";
        static const char* const back_help_en = "The Decks back buttons repeat the face buttons, so you can jump, attack and use powers without taking your thumb off the stick. The games icons stay the face ones.";
        v.push_back({Kind::Fixed, "L4", pt ? back_help_pt : back_help_en, nullptr, nullptr, 0, nullptr, TL("Same as A", "Igual ao A")});
        v.push_back({Kind::Fixed, "L5", pt ? back_help_pt : back_help_en, nullptr, nullptr, 0, nullptr, TL("Same as B", "Igual ao B")});
        v.push_back({Kind::Fixed, "R4", pt ? back_help_pt : back_help_en, nullptr, nullptr, 0, nullptr, TL("Same as X", "Igual ao X")});
        v.push_back({Kind::Fixed, "R5", pt ? back_help_pt : back_help_en, nullptr, nullptr, 0, nullptr, TL("Same as Y", "Igual ao Y")});
        v.push_back({Kind::Section, TL("Buttons", "Botões")});
        v.push_back({Kind::Fixed, TL("View button", "Botão View"),
                     TL("Opens and closes this menu with the game frozen.", "Abre e fecha este menu com o jogo congelado."),
                     nullptr, nullptr, 0, nullptr, TL("Port menu", "Menu do port")});
        v.push_back({Kind::Fixed, TL("Steam button", "Botão Steam"),
                     TL("With the Steam menu open the game stays still, like the PS button on a PS4.",
                        "Com o menu da Steam aberto o jogo fica parado, como o botão PS no PS4."),
                     nullptr, nullptr, 0, nullptr, TL("Freezes the game", "Congela o jogo")});
    } else if (which == 2) {
        // steam-deck: recursos que nao fazem parte do jogo original; vem desligados.
        v.push_back({Kind::Section, TL("Experimental", "Experimental")});
        v.push_back({Kind::Switch, TL("Skip videos", "Pular vídeos"),
                     TL("During a cutscene video, hold the menu button for 1 s to skip it; a quick tap still pauses. "
                        "The rest of the scene's audio and subtitles may still play for a few seconds.",
                        "Durante um vídeo de cena, segure o botão de menu por 1 s para pular; um toque rápido continua "
                        "pausando. O resto do áudio e das legendas da cena pode tocar por alguns segundos."),
                     nullptr, nullptr, 0, &s.movie_skip});
        v.push_back({Kind::Section, "Easter Eggs"});
        v.push_back({Kind::Switch, "Easter Eggs",
                     TL("Some graffiti of the spray minigame get a surprise. Applies to the graffiti loaded after the "
                        "change; restart the game to be sure.",
                        "Alguns grafites do minigame de spray ganham uma surpresa. Vale para os grafites carregados "
                        "depois da mudança; reinicie o jogo para garantir."),
                     nullptr, nullptr, 0, &s.easter_eggs});
    } else {
        v.push_back({Kind::Section, "Port"});
        v.push_back({Kind::Fixed, TL("Edition", "Edição"),
                     TL("The native inFAMOUS Second Son port tuned for the Steam Deck: 16:10 full screen, Deck controls, its own menu and performance work.",
                        "Port nativo do inFAMOUS Second Son ajustado pro Steam Deck: tela cheia 16:10, controles do Deck, menu próprio e otimizações de desempenho."),
                     nullptr, nullptr, 0, nullptr, "Steam Deck",
                     {{TL("Screen", "Tela"), TL("1280x800, no bars", "1280x800 sem barras"), 0},
                      {TL("Controls", "Controles"), TL("trackpads, gyroscope, shake", "trackpads, giroscópio, chacoalhar"), 0}}});
        v.push_back({Kind::Fixed, TL("Original port", "Port original"),
                     TL("The native port this edition builds on: iss_pc, by HayatoG. It runs the game directly on the processor, with graphics adapted from the shadPS4 emulator.",
                        "O port nativo em que esta edição se baseia: iss_pc, de HayatoG. Roda o jogo direto no processador, com a parte gráfica adaptada do emulador shadPS4."),
                     nullptr, nullptr, 0, nullptr, "iss_pc",
                     {{TL("Author", "Autor"), "HayatoG", 0}, {TL("Repository", "Repositório"), "github.com/HayatoG/iss_pc", 2},
                      {TL("Graphics base", "Base gráfica"), "shadPS4", 0}}});
        v.push_back({Kind::Section, TL("Credits", "Créditos")});
        v.push_back({Kind::Fixed, TL("Steam Deck edition", "Edição Steam Deck"),
                     TL("Steam Deck adaptation, performance, controls and this menu.",
                        "Adaptação pro Steam Deck, desempenho, controles e este menu."),
                     nullptr, nullptr, 0, nullptr, "Kewin Lima",
                     {{"Instagram", "@kewinlima", 2}, {"GitHub", "KewinBR", 2}}});
        v.push_back({Kind::Fixed, TL("Game", "Jogo"),
                     TL("inFAMOUS Second Son (2014), Sucker Punch Productions and Sony Interactive Entertainment. This port needs your own copy of the game.",
                        "inFAMOUS Second Son (2014), Sucker Punch Productions e Sony Interactive Entertainment. Este port precisa da sua própria cópia do jogo."),
                     nullptr, nullptr, 0, nullptr, "Sucker Punch"});
    }
    return v;
}

// Maiúsculas que entendem acento (UTF-8, letras latinas acentuadas: C3 A0..BE -> C3 80..9E).
std::string Upper(const char* text) {
    std::string r(text);
    for (size_t i = 0; i < r.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(r[i]);
        if (c < 0x80) {
            r[i] = char(std::toupper(c));
        } else if (c == 0xC3 && i + 1 < r.size()) {
            const unsigned char d = static_cast<unsigned char>(r[i + 1]);
            if (d >= 0xA0 && d <= 0xBE && d != 0xB7) r[i + 1] = char(d - 0x20);
            ++i;
        } else if (c >= 0xC0 && c < 0xE0) {
            ++i;
        } else if (c >= 0xE0 && c < 0xF0) {
            i += 2;
        } else if (c >= 0xF0) {
            i += 3;
        }
    }
    return r;
}

// Som de abertura (swoosh): <pasta do usuário>/sons/menu_abrir.wav ou TLG_MENU_SOUND; sem arquivo, nada.
// Botão Steam: o script de fora (congelar-no-menu-steam.py) cria TLG_STEAM_PAUSE_FILE enquanto o menu da Steam
// cobre o jogo e apaga ao voltar. O port congela as threads do jogo (como no menu do View) e escurece a tela;
// na volta, uma pausa curta, o escurecimento some e só então o jogo descongela.
bool steam_paused = false;
double steam_at = 0.0, steam_release_at = -1.0;
void SteamPause(ImDrawList* dl, float W, float H) {
    static const char* file = [] {
        const char* e = std::getenv("TLG_STEAM_PAUSE_FILE");
        return e && *e ? e : nullptr;
    }();
    if (!file) return;
    const double now = ImGui::GetTime();
    static double last_check = 0.0;
    static bool present = false;
    if (now - last_check > 0.05) {
        last_check = now;
        std::error_code ec;
        present = std::filesystem::exists(file, ec);
    }
    if (present && !steam_paused) {
        steam_paused = true;
        steam_at = now;
        steam_release_at = -1.0;
        tlggpu_set_guest_paused(1);
        runtime_pause_guest(1);
        std::puts("Overlay: Steam menu, game frozen");
    } else if (!present && steam_paused && steam_release_at < 0.0) {
        steam_release_at = now;
    } else if (present && steam_release_at >= 0.0) {
        steam_release_at = -1.0; // abriu de novo antes de voltar
        steam_at = now - 1.0;
    }
    if (!steam_paused) return;
    // a volta espera o menu da Steam terminar de fechar (~0,3 s) antes de tirar o escurecimento
    constexpr float Hold = 0.45f, Fade = 0.35f;
    float dim = std::min(1.0f, float(now - steam_at) / 0.12f);
    if (steam_release_at >= 0.0) {
        const float tr = float(now - steam_release_at);
        dim *= 1.0f - std::clamp((tr - Hold) / Fade, 0.0f, 1.0f);
        if (tr >= Hold + Fade) {
            steam_paused = false;
            runtime_pause_guest(0);
            tlggpu_set_guest_paused(0);
            std::puts("Overlay: back from the Steam menu, game resumed");
            return;
        }
    }
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), IM_COL32(118, 118, 118, u32(52 * dim)));
    dl->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(W, H), IM_COL32(5, 6, 8, u32(205 * dim)),
                                IM_COL32(5, 6, 8, u32(160 * dim)), IM_COL32(5, 6, 8, u32(180 * dim)),
                                IM_COL32(5, 6, 8, u32(205 * dim)));
}

void PlayOpenSound() {
    static SDL_AudioStream* stream = nullptr;
    static Uint8* data = nullptr;
    static Uint32 size = 0;
    static bool tried = false;
    if (!tried) {
        tried = true;
        std::string path;
        if (const char* e = std::getenv("TLG_MENU_SOUND"); e && *e) {
            path = e;
        } else if (const char* u = std::getenv("TLG_GPU_USER_DIR"); u && *u) {
            path = std::string(u) + "/sons/menu_abrir.wav";
        }
        SDL_AudioSpec spec{};
        if (!path.empty() && SDL_InitSubSystem(SDL_INIT_AUDIO) && SDL_LoadWAV(path.c_str(), &spec, &data, &size)) {
            stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
            if (stream) {
                SDL_ResumeAudioStreamDevice(stream);
            }
        }
    }
    if (stream && data) {
        SDL_ClearAudioStream(stream);
        SDL_PutAudioStreamData(stream, data, int(size));
    }
}

int FirstSelectable(const std::vector<Item>& items) {
    for (int i = 0; i < int(items.size()); ++i) {
        if (items[i].kind != Kind::Section) {
            return i;
        }
    }
    return 0;
}

// Uma mudança de valor: as regras da predefinição (como na aba antiga) e o "pop".
void Change(Item& it, int index, int step) {
    auto& s = TlgSettings::Get();
    const double now = ImGui::GetTime();
    if (it.kind == Kind::Switch && it.flag) {
        *it.flag = !it.flag->load();
        switch_changed[tab][index] = now;
    } else if (it.kind == Kind::Choice && it.choice) {
        *it.choice = (it.choice->load() + step + it.count) % it.count;
        value_changed[tab][index] = now;
    } else if (it.kind == Kind::Action && it.id == 10) {
        classic = true;
        return;
    } else {
        return;
    }
    dirty = true;
    if (it.id == 1 && s.gfx_preset < 3) {
        const int p = s.gfx_preset;
        s.gfx_resolution = p == 0 ? 0 : 2;
        s.gfx_distance = p;
        s.gfx_shadows = p;
        s.particles = p;
        s.gfx_motion_blur = p == 0;
        s.gfx_ssr = p < 2;
        for (auto& t : value_changed[tab]) t = now;
    } else if (it.id == 2 && s.gfx_preset < 3) {
        s.gfx_preset = 3; // um item só mudou: personalizado
        value_changed[tab][1] = now;
    }
}

void Text(ImDrawList* dl, ImFont* font, float size, ImVec2 pos, u32 color, const char* text, float wrap = 0.0f) {
    dl->AddText(font, size, pos, color, text, nullptr, wrap);
}
ImVec2 Measure(ImFont* font, float size, const char* text) {
    return font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
}

void Draw() {
    ImGuiIO& io = ImGui::GetIO();
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float W = vp->Size.x, H = vp->Size.y;
    const float u = W / 100.0f; // 1 cqw da prévia
    ImFont* reg = font_regular ? font_regular : ImGui::GetFont();
    ImFont* bold = font_bold ? font_bold : reg;
    const double now = ImGui::GetTime();
    static double last_draw = -10.0;
    if (!shown || now - last_draw > 0.25) {
        shown = true;
        opened_at = now;
        tab_changed_at = now - 10.5;
        PlayOpenSound();
        snap = Take();
        dialog = 0;
        close_request = false;
        closing_at = -1.0;
        // Testes: TLG_MENU_TAB=n abre o menu nessa aba (print do menu sem ninguem no controle).
        if (const char* p = std::getenv("TLG_MENU_TAB"); p && *p) {
            tab = std::clamp(std::atoi(p), 0, 3);
            sel = -1;
        }
    }
    last_draw = now;
    io.MouseDrawCursor = false;
    const float t = float(now - opened_at);
    const bool pt = UiPortuguese();

    // Entrada: painéis 0,55 s, topo e rodapé 0,45 s, conteúdo depois de 0,3 s.
    const float e_list = Ease(t / 0.275f), e_ctx = Ease((t - 0.04f) / 0.275f);
    const float e_top = Ease(t / 0.225f), e_bot = Ease((t - 0.025f) / 0.225f);
    const float fade = std::clamp((t - 0.15f) / 0.15f, 0.0f, 1.0f);

    // Entrada do teclado/controle.
    static const char* const tab_names_pt[] = {"Gráficos", "Controle", "Extras", "Sobre"};
    static const char* const tab_names_en[] = {"Graphics", "Controls", "Extras", "About"};
    const char* const* tab_names = pt ? tab_names_pt : tab_names_en;
    constexpr int TabCount = 4;
    auto items = Build(tab);
    if (sel < 0 || sel >= int(items.size()) || items[sel].kind == Kind::Section) {
        sel = FirstSelectable(items);
    }
    const auto pressed = [](ImGuiKey a, ImGuiKey b = ImGuiKey_None, bool repeat = true) {
        return ImGui::IsKeyPressed(a, repeat) || (b != ImGuiKey_None && ImGui::IsKeyPressed(b, repeat));
    };
    const auto go_tab = [&](int step) {
        prev_tab = tab;
        tab = (tab + step + TabCount) % TabCount;
        tab_changed_at = now;
        (step > 0 ? r1_at : l1_at) = now;
        items = Build(tab);
        sel = FirstSelectable(items);
        scroll = scroll_target = 0.0f;
    };
    const bool want_close = close_request.exchange(false) || pressed(ImGuiKey_GamepadFaceRight, ImGuiKey_Escape, false);
    const auto finish = [&] {
        dialog = 0;
        if (closing_at < 0.0) closing_at = now;
    };
    // Fechando: 0,18 s o menu some, 0,15 s de pausa, 0,25 s o escurecimento some; só então o menu fecha e o
    // jogo descongela (a thread do controle espera o menu fechar).
    constexpr float FadeMenu = 0.18f, Hold = 0.15f, FadeDim = 0.25f;
    const float tc = closing_at < 0.0 ? 0.0f : float(now - closing_at);
    if (closing_at >= 0.0 && tc >= FadeMenu + Hold + FadeDim) {
        closing_at = -1.0;
        SetOpen(false);
        return;
    }
    const float menu_alpha = closing_at < 0.0 ? 1.0f : 1.0f - std::clamp(tc / FadeMenu, 0.0f, 1.0f);
    const float dim_alpha = closing_at < 0.0 ? 1.0f : 1.0f - std::clamp((tc - FadeMenu - Hold) / FadeDim, 0.0f, 1.0f);
    if (closing_at >= 0.0) {
        close_request = false; // fechando: nada mais responde
    }
    const auto open_dialog = [&](int which) {
        dialog = which;
        dialog_sel = 0;
        dialog_at = now;
    };
    if (closing_at >= 0.0) {
        // ignora a entrada
    } else if (dialog == 0 && want_close) {
        if (Changed(false)) {
            open_dialog(1);
        } else {
            finish();
        }
    } else if (dialog) {
        const bool lr = pressed(ImGuiKey_GamepadDpadLeft, ImGuiKey_LeftArrow) || pressed(ImGuiKey_GamepadDpadRight, ImGuiKey_RightArrow) ||
                        pressed(ImGuiKey_GamepadDpadUp, ImGuiKey_UpArrow) || pressed(ImGuiKey_GamepadDpadDown, ImGuiKey_DownArrow);
        if (lr && dialog != 3) {
            dialog_sel ^= 1;
        }
        const bool ok = pressed(ImGuiKey_GamepadFaceDown, ImGuiKey_Enter, false);
        if (dialog == 1) {
            if (want_close) {
                dialog = 0; // B: volta pro menu
            } else if (ok && dialog_sel == 1) { // Descartar
                Restore(snap);
                dirty = false;
                finish();
            } else if (ok) { // Salvar
                const bool restart = Changed(true);
                dirty = true;
                TlgSettings::Save();
                dirty = false;
                if (!restart) {
                    finish();
                }
                open_dialog(CanRestart() ? 2 : 3);
            }
        } else if (dialog == 2) {
            if (want_close || (ok && dialog_sel == 1)) { // Depois
                finish();
            }
            if (ok) {
                std::puts("Settings: restarting the game to apply the graphics options");
                runtime_restart();
            }
        } else if (dialog == 3 && (ok || want_close || now - dialog_at > 4.0)) {
            finish();
        }
    }
    const bool menu_input = dialog == 0 && closing_at < 0.0;
    if (menu_input && pressed(ImGuiKey_GamepadR1, ImGuiKey_E, false)) go_tab(1);
    if (menu_input && pressed(ImGuiKey_GamepadL1, ImGuiKey_Q, false)) go_tab(-1);
    const auto move = [&](int step) {
        int k = sel;
        do {
            k = (k + step + int(items.size())) % int(items.size());
        } while (items[k].kind == Kind::Section);
        sel = k;
        sel_changed_at = now;
    };
    if (menu_input) {
        if (pressed(ImGuiKey_GamepadDpadDown, ImGuiKey_DownArrow)) move(1);
        if (pressed(ImGuiKey_GamepadDpadUp, ImGuiKey_UpArrow)) move(-1);
        if (pressed(ImGuiKey_GamepadDpadRight, ImGuiKey_RightArrow)) Change(items[sel], sel, 1);
        if (pressed(ImGuiKey_GamepadDpadLeft, ImGuiKey_LeftArrow)) Change(items[sel], sel, -1);
        if (pressed(ImGuiKey_GamepadFaceDown, ImGuiKey_Enter, false)) Change(items[sel], sel, 1);
    }

    ImDrawList* dl = ImGui::GetForegroundDrawList();
    // Fundo: o jogo congelado escurecido (mais à esquerda, onde fica a lista).
    const float dim = std::min(1.0f, t / 0.12f) * dim_alpha;
    dl->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), IM_COL32(118, 118, 118, u32(52 * dim)));
    dl->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(W, H), IM_COL32(5, 6, 8, u32(205 * dim)),
                                IM_COL32(5, 6, 8, u32(160 * dim)), IM_COL32(5, 6, 8, u32(180 * dim)),
                                IM_COL32(5, 6, 8, u32(205 * dim)));
    const int menu_vtx0 = dl->VtxBuffer.Size; // tudo daqui pra frente apaga junto no fechamento

    // ---- topo: L1, abas, R1 ----
    {
        const float h = 4.4f * u, y = 3.3f * u - (1.0f - e_top) * (h + 3.3f * u);
        const float key_w = 5.4f * u, gap = 0.5f * u, x0 = 1.6f * u, x1 = W - 1.6f * u;
        const auto key = [&](float x, const char* label, double at) {
            dl->AddRectFilled(ImVec2(x, y), ImVec2(x + key_w, y + h), Faixa);
            const float flash = 1.0f - std::clamp(float(now - at) / 0.25f, 0.0f, 1.0f);
            const float sz = 1.9f * u * (1.0f + 0.15f * flash);
            const ImVec2 ts = Measure(bold, sz, label);
            const ImVec2 c(x + key_w * 0.5f, y + h * 0.5f);
            dl->AddRectFilled(ImVec2(c.x - ts.x * 0.5f - 0.45f * u, c.y - ts.y * 0.5f - 0.1f * u),
                              ImVec2(c.x + ts.x * 0.5f + 0.45f * u, c.y + ts.y * 0.5f + 0.1f * u),
                              Mix(Branco, Laranja, flash), 0.25f * u);
            Text(dl, bold, sz, ImVec2(c.x - ts.x * 0.5f, c.y - ts.y * 0.5f), IM_COL32(17, 17, 17, 255), label);
        };
        key(x0, "L1", l1_at);
        key(x1 - key_w, "R1", r1_at);
        dl->AddRectFilled(ImVec2(x0 + key_w + gap, y), ImVec2(x1 - key_w - gap, y + h), Faixa);
        float x = x0 + key_w + gap + 2.0f * u;
        const float k_tab = Ease(float(now - tab_changed_at) / 0.35f);
        for (int i = 0; i < TabCount; ++i) {
            float grow = i == tab ? k_tab : i == prev_tab && prev_tab != tab ? 1.0f - k_tab : 0.0f;
            if (now - tab_changed_at > 10.0) grow = i == tab ? 1.0f : 0.0f;
            const float sz = (1.75f + (3.1f - 1.75f) * grow) * u;
            ImFont* f = grow > 0.5f ? bold : reg;
            const ImVec2 ts = Measure(f, sz, tab_names[i]);
            Text(dl, f, sz, ImVec2(x, y + (h - ts.y) * 0.5f), Mix(Apagado, Amarelo, grow), tab_names[i]);
            x += ts.x + 2.6f * u;
        }
    }

    // ---- lista ----
    const float lx0 = 7.4f * u, ly0 = 10.6f * u, lw = 42.5f * u, ly1 = H - 8.6f * u;
    {
        const float cur_x0 = lx0 + lw * (1.0f - e_list); // cresce da direita pra esquerda
        const float a = 0.2f + 0.8f * e_list;
        dl->AddRectFilled(ImVec2(cur_x0, ly0), ImVec2(lx0 + lw, ly1), Alpha(Painel, a));
        dl->AddRect(ImVec2(cur_x0, ly0), ImVec2(lx0 + lw, ly1), Alpha(Branco, a), 0.0f, 0, 0.12f * u);
        const float n = 0.7f * u;
        dl->AddRectFilled(ImVec2(cur_x0 - 0.4f * u, ly0 - 0.4f * u), ImVec2(cur_x0 - 0.4f * u + n, ly0 - 0.4f * u + n), Alpha(Branco, a));
        dl->AddRectFilled(ImVec2(cur_x0 - 0.4f * u, ly1 - 0.3f * u), ImVec2(cur_x0 - 0.4f * u + n, ly1 - 0.3f * u + n), Alpha(Branco, a));
        dl->AddRectFilled(ImVec2(lx0 + lw - 9.0f * u * e_list, ly0 - 0.5f * u), ImVec2(lx0 + lw, ly0), Alpha(Branco, a));

        const float px = lx0 + 2.4f * u, py = ly0 + 1.7f * u;
        const char* title = tab == 0 ? TL("Options", "Opções") : tab_names[tab];
        std::string up = title;
        up = Upper(up.c_str());
        if (tab == 0) up = TL("OPTIONS", "OPÇÕES");
        else if (tab == 1) up = TL("CONTROLS", "CONTROLE");
        else if (tab == 2) up = "EXTRAS";
        else up = TL("ABOUT", "SOBRE");
        Text(dl, bold, 5.4f * u, ImVec2(px, py), Alpha(Amarelo, fade), up.c_str());

        // itens, com rolagem que acompanha a seleção
        const float area_y0 = py + 6.6f * u, area_y1 = ly1 - 1.4f * u, area_h = area_y1 - area_y0;
        std::vector<float> ys(items.size());
        float yy = 0.0f;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].kind == Kind::Section && i) yy += 1.1f * u;
            ys[i] = yy;
            yy += items[i].kind == Kind::Section ? 3.0f * u : 2.6f * u;
        }
        const float total = yy;
        if (ys[sel] - scroll_target < 0.0f) scroll_target = ys[sel] - (sel > 0 && items[sel - 1].kind == Kind::Section ? 3.4f * u : 0.0f);
        if (ys[sel] + 2.6f * u - scroll_target > area_h) scroll_target = ys[sel] + 2.6f * u - area_h;
        scroll_target = std::clamp(scroll_target, 0.0f, std::max(0.0f, total - area_h));
        scroll += (scroll_target - scroll) * std::min(1.0f, io.DeltaTime * 14.0f);

        dl->PushClipRect(ImVec2(lx0, area_y0 - 0.2f * u), ImVec2(lx0 + lw, area_y1 + 0.2f * u), true);
        const float val_x1 = lx0 + lw - 4.0f * u; // coluna do valor: setas fixas
        const float col_w = 13.5f * u, arrow_w = 1.4f * u;
        for (size_t i = 0; i < items.size(); ++i) {
            const Item& it = items[i];
            const float y = area_y0 + ys[i] - scroll;
            if (y > area_y1 + 3.0f * u || y < area_y0 - 4.0f * u) continue;
            if (it.kind == Kind::Section) {
                std::string sec = it.label;
                sec = Upper(sec.c_str());
                Text(dl, bold, 2.6f * u, ImVec2(px, y), Alpha(Branco, fade), sec.c_str());
                continue;
            }
            const bool on = int(i) == sel;
            const u32 col = Alpha(on ? Amarelo : Texto, fade);
            const float fs = 1.75f * u;
            Text(dl, reg, fs, ImVec2(px + 1.9f * u, y + 0.35f * u), col, it.label);
            const float cy = y + 0.35f * u + fs * 0.6f;
            const float pop = 1.0f + 0.35f * (1.0f - Ease(float(now - value_changed[tab][i]) / 0.32f));
            if (it.kind == Kind::Choice) {
                const char* val = it.names[std::clamp(it.choice->load(), 0, it.count - 1)];
                const float sz = fs * pop;
                const ImVec2 ts = Measure(reg, sz, val);
                const float cx = val_x1 - arrow_w - col_w * 0.5f;
                Text(dl, reg, sz, ImVec2(cx - ts.x * 0.5f, cy - ts.y * 0.5f), col, val);
                if (on) {
                    const float ah = 0.55f * u, ax0 = val_x1 - arrow_w - col_w - arrow_w * 0.5f, ax1 = val_x1 - arrow_w * 0.5f;
                    dl->AddTriangleFilled(ImVec2(ax0 - ah * 0.5f, cy), ImVec2(ax0 + ah * 0.5f, cy - ah), ImVec2(ax0 + ah * 0.5f, cy + ah), col);
                    dl->AddTriangleFilled(ImVec2(ax1 + ah * 0.5f, cy), ImVec2(ax1 - ah * 0.5f, cy - ah), ImVec2(ax1 - ah * 0.5f, cy + ah), col);
                }
            } else if (it.kind == Kind::Switch) {
                const bool v = it.flag->load();
                const float k = Ease(float(now - switch_changed[tab][i]) / 0.28f);
                const float pos = v ? k : 1.0f - k; // 0 desligado, 1 ligado
                const float sw = 3.4f * u, sh = 1.6f * u, sx = val_x1 - arrow_w - col_w * 0.5f - sw * 0.5f, sy = cy - sh * 0.5f;
                dl->AddRect(ImVec2(sx, sy), ImVec2(sx + sw, sy + sh), col, 0.0f, 0, 0.12f * u);
                const float pad = 0.2f * u, kw = sw * 0.5f - pad;
                const float kx = sx + pad + pos * (sw - 2.0f * pad - kw);
                dl->AddRectFilled(ImVec2(kx, sy + pad), ImVec2(kx + kw, sy + sh - pad), Alpha(on ? Amarelo : Texto, fade * (0.3f + 0.7f * pos)));
            } else {
                const ImVec2 ts = Measure(reg, fs, it.fixed);
                Text(dl, reg, fs, ImVec2(val_x1 - ts.x, cy - ts.y * 0.5f), col, it.fixed);
            }
        }
        dl->PopClipRect();
        if (total > area_h + 1.0f) { // barra de rolagem
            const float bx = lx0 + lw - 0.9f * u, by0 = area_y0, by1 = area_y1;
            dl->AddRectFilled(ImVec2(bx, by0), ImVec2(bx + 0.12f * u, by1), Alpha(IM_COL32(233, 231, 225, 46), fade));
            const float th = std::max(3.0f * u, (by1 - by0) * area_h / total);
            const float ty = by0 + (by1 - by0 - th) * (scroll / std::max(1.0f, total - area_h));
            dl->AddRectFilled(ImVec2(bx - 0.35f * u, ty), ImVec2(bx + 0.45f * u, ty + th), Alpha(IM_COL32(233, 231, 225, 150), fade));
        }
    }

    // ---- contexto ----
    {
        const Item& it = items[sel];
        const float cx0 = 54.4f * u, cy0 = 15.4f * u, cw = 37.6f * u, pad = 2.2f * u;
        const float inner = cw - 2.0f * pad;
        const float k_sel = Ease(float(now - std::max(sel_changed_at, tab_changed_at)) / 0.25f);
        const float a_txt = fade * k_sel, dx = (1.0f - k_sel) * -0.8f * u;
        float y = cy0 + 1.8f * u;
        const float title_sz = 3.6f * u, help_sz = 1.6f * u;
        std::string tt = it.label;
        tt = Upper(tt.c_str());
        const ImVec2 title_ts = bold->CalcTextSizeA(title_sz, FLT_MAX, inner, tt.c_str());
        const ImVec2 help_ts = reg->CalcTextSizeA(help_sz, FLT_MAX, inner, it.help);
        float h = 1.8f * u + 1.3f * u + 0.5f * u + title_ts.y + 1.1f * u + help_ts.y + 2.0f * u;
        if (!it.effects.empty()) h += 1.6f * u + 1.2f * u + float(it.effects.size()) * 2.0f * u;
        const float cur_x1 = cx0 + cw * e_ctx; // cresce da esquerda pra direita
        const float a = 0.2f + 0.8f * e_ctx;
        dl->AddRectFilled(ImVec2(cx0, cy0), ImVec2(cur_x1, cy0 + h), Alpha(Painel, a));
        dl->AddRect(ImVec2(cx0, cy0), ImVec2(cur_x1, cy0 + h), Alpha(Branco, a), 0.0f, 0, 0.12f * u);
        const float n = 0.7f * u;
        dl->AddRectFilled(ImVec2(cx0 - 0.4f * u, cy0 - 0.4f * u), ImVec2(cx0 - 0.4f * u + n, cy0 - 0.4f * u + n), Alpha(Branco, a));
        dl->AddRectFilled(ImVec2(cur_x1 - 0.3f * u, cy0 - 0.4f * u), ImVec2(cur_x1 - 0.3f * u + n, cy0 - 0.4f * u + n), Alpha(Branco, a));
        dl->PushClipRect(ImVec2(cx0, cy0 - u), ImVec2(cur_x1, cy0 + h + u), true);
        std::string rot = tab_names[tab];
        rot = Upper(rot.c_str());
        Text(dl, reg, 1.25f * u, ImVec2(cx0 + pad, y), Alpha(Apagado, fade), rot.c_str());
        y += 1.3f * u + 0.5f * u;
        Text(dl, bold, title_sz, ImVec2(cx0 + pad + dx, y), Alpha(Amarelo, a_txt), tt.c_str(), inner);
        y += title_ts.y + 1.1f * u;
        Text(dl, reg, help_sz, ImVec2(cx0 + pad + dx, y), Alpha(Texto, a_txt), it.help, inner);
        y += help_ts.y;
        if (!it.effects.empty()) {
            y += 1.6f * u;
            dl->AddLine(ImVec2(cx0 + pad, y), ImVec2(cx0 + cw - pad, y), Alpha(IM_COL32(233, 231, 225, 64), fade), 0.1f * u);
            y += 1.2f * u;
            for (const auto& ef : it.effects) {
                std::string lab = ef.label;
                lab = Upper(lab.c_str());
                Text(dl, reg, 1.2f * u, ImVec2(cx0 + pad + dx, y + 0.15f * u), Alpha(Apagado, a_txt), lab.c_str());
                const u32 vc = ef.tone == 1 ? Verde : ef.tone == 2 ? Amarelo : Branco;
                Text(dl, reg, 1.5f * u, ImVec2(cx0 + pad + 11.0f * u + dx, y), Alpha(vc, a_txt), ef.value);
                y += 2.0f * u;
            }
        }
        dl->PopClipRect();
        // selo
        const char* selo = TL("STEAM DECK  ·  1280x800  ·  GAME FROZEN", "STEAM DECK  ·  1280x800  ·  JOGO CONGELADO");
        const ImVec2 ss = Measure(reg, 1.2f * u, selo);
        Text(dl, reg, 1.2f * u, ImVec2(W - 8.0f * u - ss.x, 46.0f * u), Alpha(Apagado, fade), selo);
    }

    // ---- rodapé ----
    {
        const float h = 3.7f * u, y = H - 2.2f * u - h + (1.0f - e_bot) * (h + 2.2f * u) * 1.6f;
        const float bs = 2.1f * u, fs = 1.75f * u;
        const auto hint = [&](float x_right, const char* label, int kind) {
            const ImVec2 ts = Measure(reg, fs, label);
            const float w = 1.6f * u + bs + 0.8f * u + ts.x + 1.6f * u + (kind == 2 ? 0.5f * u : 0.0f);
            const float x0 = x_right - w;
            dl->AddRectFilled(ImVec2(x0, y), ImVec2(x_right, y + h), Faixa);
            const ImVec2 c(x0 + 1.6f * u + bs * 0.5f + (kind == 2 ? 0.25f * u : 0.0f), y + h * 0.5f);
            if (kind == 2) { // View: dois retângulos sobrepostos
                dl->AddRectFilled(ImVec2(c.x - 1.3f * u, c.y - 1.05f * u), ImVec2(c.x + 1.3f * u, c.y + 1.05f * u), Branco, 0.35f * u);
                const u32 k = IM_COL32(17, 17, 17, 255);
                dl->AddRect(ImVec2(c.x - 0.75f * u, c.y - 0.2f * u), ImVec2(c.x + 0.35f * u, c.y + 0.65f * u), k, 0.0f, 0, 0.16f * u);
                dl->AddLine(ImVec2(c.x - 0.35f * u, c.y - 0.55f * u), ImVec2(c.x + 0.75f * u, c.y - 0.55f * u), k, 0.16f * u);
                dl->AddLine(ImVec2(c.x + 0.75f * u, c.y - 0.55f * u), ImVec2(c.x + 0.75f * u, c.y + 0.3f * u), k, 0.16f * u);
            } else {
                dl->AddCircleFilled(c, bs * 0.5f, Branco, 24);
                const char* letter = kind == 0 ? "A" : "B";
                const ImVec2 ls = Measure(bold, 1.35f * u, letter);
                Text(dl, bold, 1.35f * u, ImVec2(c.x - ls.x * 0.5f, c.y - ls.y * 0.5f), IM_COL32(17, 17, 17, 255), letter);
            }
            Text(dl, reg, fs, ImVec2(x0 + 1.6f * u + bs + 0.8f * u + (kind == 2 ? 0.5f * u : 0.0f), y + (h - ts.y) * 0.5f), Branco, label);
            return x0;
        };
        const float right = W - 1.6f * u;
        const float xb = hint(right, TL("Back", "Voltar"), 1);
        hint(xb - 0.5f * u, TL("Select", "Selecionar"), 0);
        // View à esquerda
        const char* fechar = TL("Close", "Fechar");
        const ImVec2 ts = Measure(reg, fs, fechar);
        hint(1.6f * u + 1.6f * u + bs + 0.8f * u + ts.x + 1.6f * u + 0.5f * u, fechar, 2);
    }

    // ---- diálogo de fechar (salvar / reiniciar), no estilo dos painéis ----
    if (dialog) {
        const float k = Ease(float(now - dialog_at) / 0.15f);
        dl->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), IM_COL32(0, 0, 0, u32(110 * k)));
        const float dw = 46.0f * u, dh = 21.0f * u, cx = W * 0.5f, cy = H * 0.5f;
        const float hw = dw * 0.5f * k; // cresce do centro pros lados
        const ImVec2 a(cx - hw, cy - dh * 0.5f), b(cx + hw, cy + dh * 0.5f);
        dl->AddRectFilled(a, b, IM_COL32(14, 15, 18, 245));
        dl->AddRect(a, b, Branco, 0.0f, 0, 0.12f * u);
        const float n = 0.7f * u;
        dl->AddRectFilled(ImVec2(a.x - 0.4f * u, a.y - 0.4f * u), ImVec2(a.x - 0.4f * u + n, a.y - 0.4f * u + n), Branco);
        dl->AddRectFilled(ImVec2(b.x - 0.3f * u, b.y - 0.3f * u), ImVec2(b.x - 0.3f * u + n, b.y - 0.3f * u + n), Branco);
        dl->AddRectFilled(ImVec2(b.x - 9.0f * u * k, a.y - 0.5f * u), ImVec2(b.x, a.y), Branco);
        const float x0 = cx - dw * 0.5f + 2.6f * u, inner = dw - 5.2f * u;
        const float fa = std::clamp((float(now - dialog_at) - 0.08f) / 0.12f, 0.0f, 1.0f);
        const char* title = dialog == 1 ? TL("SAVE THE CHANGES?", "SALVAR AS ALTERAÇÕES?")
                          : TL("RESTART TO APPLY", "REINICIE PARA APLICAR");
        const char* body = dialog == 1
            ? TL("You changed options in this menu. Save them, or go back to how they were when it opened.",
                 "Você mudou opções neste menu. Salve, ou volte para como estavam quando ele abriu.")
            : dialog == 2
            ? TL("Graphics options apply when the game opens. Restart now? The game continues from the last save.",
                 "As opções gráficas valem quando o jogo abre. Reiniciar agora? O jogo volta do último salvamento.")
            : TL("Saved. Graphics options apply the next time the game opens.",
                 "Salvo. As opções gráficas valem na próxima vez que o jogo abrir.");
        dl->PushClipRect(a, b, true);
        Text(dl, bold, 3.4f * u, ImVec2(x0, a.y + 2.0f * u), Alpha(Amarelo, fa), title);
        Text(dl, reg, 1.65f * u, ImVec2(x0, a.y + 6.6f * u), Alpha(Texto, fa), body, x0 + inner);
        if (dialog != 3) {
            const char* opt[2] = {dialog == 1 ? TL("Save", "Salvar") : TL("Restart now", "Reiniciar agora"),
                                  dialog == 1 ? TL("Discard", "Descartar") : TL("Later", "Depois")};
            const float bw = 15.0f * u, bh = 3.6f * u, by = b.y - 2.4f * u - bh;
            for (int i = 0; i < 2; ++i) {
                const float bx = x0 + i * (bw + 1.4f * u);
                const bool on = dialog_sel == i;
                dl->AddRectFilled(ImVec2(bx, by), ImVec2(bx + bw, by + bh), on ? Alpha(Amarelo, fa) : IM_COL32(0, 0, 0, u32(150 * fa)));
                if (!on) dl->AddRect(ImVec2(bx, by), ImVec2(bx + bw, by + bh), Alpha(IM_COL32(233, 231, 225, 120), fa), 0.0f, 0, 0.1f * u);
                const ImVec2 ts = Measure(bold, 1.75f * u, opt[i]);
                Text(dl, bold, 1.75f * u, ImVec2(bx + (bw - ts.x) * 0.5f, by + (bh - ts.y) * 0.5f),
                     on ? IM_COL32(17, 17, 17, u32(255 * fa)) : Alpha(Branco, fa), opt[i]);
            }
        } else {
            // contagem pra voltar ao jogo
            const float left = std::clamp(1.0f - float(now - dialog_at) / 4.0f, 0.0f, 1.0f);
            const float by = b.y - 3.0f * u;
            dl->AddRectFilled(ImVec2(x0, by), ImVec2(x0 + inner, by + 0.35f * u), IM_COL32(233, 231, 225, u32(50 * fa)));
            dl->AddRectFilled(ImVec2(x0, by), ImVec2(x0 + inner * left, by + 0.35f * u), Alpha(Amarelo, fa));
        }
        dl->PopClipRect();
    }
    if (menu_alpha < 1.0f) { // fade do menu inteiro (vértices já gravados)
        for (int i = menu_vtx0; i < dl->VtxBuffer.Size; ++i) {
            ImU32& c = dl->VtxBuffer[i].col;
            const u32 a = u32(float((c >> IM_COL32_A_SHIFT) & 0xFF) * menu_alpha);
            c = (c & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
        }
    }
}

void LoadFonts(ImGuiIO& io) {
    const char* custom = std::getenv("TLG_MENU_FONT");
    const char* custom_bold = std::getenv("TLG_MENU_FONT_BOLD");
    std::error_code ec;
    const auto pick = [&](const char* env, const char* fallback) -> std::string {
        if (env && *env && std::filesystem::exists(env, ec)) return env;
        return std::filesystem::exists(fallback, ec) ? fallback : "";
    };
    // TLG_MENU_FONT / TLG_MENU_FONT_BOLD: ISS-Texto.ttf e ISS-Titulo.ttf (fonte_jogo_ttf.py, da fonte do jogo)
    const std::string reg = pick(custom, "/usr/share/fonts/TTF/DejaVuSansCondensed.ttf");
    const std::string bol = pick(custom_bold, "/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf");
    if (!reg.empty()) font_regular = io.Fonts->AddFontFromFileTTF(reg.c_str(), 18.0f);
    if (!bol.empty()) font_bold = io.Fonts->AddFontFromFileTTF(bol.c_str(), 18.0f);
}

} // namespace DeckMenuImpl

#ifdef TLG_RENDERER_V2
// ---- steam-deck: a tela de abertura do renderer v2 (carregamento de shaders) ----
// Aprovada pelo Kewin em 2026-10-22 (previa em Drive Steamdeck/infamous/arte-loading). Fundo: TLG_LOADING_ART (arte
// do Delsin com o Deck) cobrindo a tela, fade in de 0,9 s e zoom lento sem fim. Logo: TLG_LOADING_LOGO (o
// sce_sys/icon0.png do jogo instalado; o fundo preto e os corvos cinza saem aqui), 40% da largura, pintado como spray
// (cada linha da esquerda pra direita, borda irregular e respingos na frente) e a tinta do O de SON escorrendo ate
// atras da palavra SHADERS. Titulo, barra e tempo entram juntos da esquerda depois do logo.
namespace StartupImpl {

std::atomic<bool> active{false};
std::atomic<int> state{0};
std::atomic<u32> done{0}, total{0};
std::chrono::steady_clock::time_point started{}, state_started{};
int shown_state = -1;
bool tried = false;
ImTextureData* art = nullptr;
ImTextureData* logo = nullptr;
std::vector<float> edge;               // borda do spray por faixa (pixels a 800 de altura)
std::vector<std::array<float, 3>> dots; // respingos: x, y (fracao do logo), raio

ImTextureData* MakeTexture(const TlgPng::Image& image) {
    ImTextureData* tex = IM_NEW(ImTextureData)();
    tex->Create(ImTextureFormat_RGBA32, image.width, image.height);
    std::memcpy(tex->GetPixels(), image.rgba.data(), image.rgba.size());
    ImGui::RegisterUserTexture(tex);
    return tex;
}

// imgui_mutex held, before NewFrame.
void LoadTextures() {
    if (tried) return;
    tried = true;
    TlgPng::Image image;
    if (const char* p = std::getenv("TLG_LOADING_ART"); p && *p && TlgPng::LoadRgba(p, image))
        art = MakeTexture(image);
    if (const char* p = std::getenv("TLG_LOADING_LOGO"); p && *p && TlgPng::LoadRgba(p, image)) {
        // Fundo preto e corvos cinza fora: alfa pelo canal mais claro; recorte no que sobrou.
        int x0 = image.width, y0 = image.height, x1 = -1, y1 = -1;
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x) {
                unsigned char* px = &image.rgba[(size_t(y) * image.width + x) * 4];
                const int m = std::max({px[0], px[1], px[2]});
                px[3] = (unsigned char)std::clamp((m - 95) * 3, 0, 255);
                if (px[3] > 128) {
                    x0 = std::min(x0, x); x1 = std::max(x1, x);
                    y0 = std::min(y0, y); y1 = std::max(y1, y);
                }
            }
        if (x1 > x0 && y1 > y0) {
            TlgPng::Image crop;
            crop.width = x1 - x0 + 1;
            crop.height = y1 - y0 + 1;
            crop.rgba.resize(size_t(crop.width) * crop.height * 4);
            for (int y = 0; y < crop.height; ++y)
                std::memcpy(&crop.rgba[size_t(y) * crop.width * 4],
                            &image.rgba[(size_t(y + y0) * image.width + x0) * 4], size_t(crop.width) * 4);
            logo = MakeTexture(crop);
        }
    }
    std::printf("Overlay: tela de abertura (arte %s, logo %s)\n", art ? "ok" : "ausente", logo ? "ok" : "ausente");
    // Ruido fixo: borda do spray suavizada e os respingos.
    unsigned seed = 7;
    const auto rnd = [&seed] { seed = seed * 1103515245u + 12345u; return float((seed >> 8) & 0xFFFF) / 65535.0f; };
    edge.resize(256);
    for (float& e : edge) e = (rnd() * 2.0f - 1.0f) * 14.0f;
    for (int pass = 0; pass < 3; ++pass)
        for (size_t i = 0; i < edge.size(); ++i)
            edge[i] = (edge[i ? i - 1 : 0] + edge[i] + edge[std::min(i + 1, edge.size() - 1)]) / 3.0f;
    dots.resize(900);
    for (auto& d : dots) d = {rnd(), rnd(), 1.2f + rnd() * 2.3f};
}

// imgui_mutex held: quando a tela some, as texturas sao liberadas (a arte tem uns 4 MB).
void Retire() {
    for (ImTextureData** t : {&art, &logo})
        if (*t) {
            (*t)->WantDestroyNextFrame = true;
            choice_retired.push_back(*t);
            *t = nullptr;
        }
    choice_retiring = !choice_retired.empty();
}

float EaseOut(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return 1.0f - (1.0f - t) * (1.0f - t) * (1.0f - t);
}

u32 Col(int r, int g, int b, float a) {
    return IM_COL32(r, g, b, int(std::clamp(a, 0.0f, 1.0f) * 255.0f));
}

void Draw() {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float W = vp->Size.x, H = vp->Size.y, s = H / 800.0f;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const auto now = std::chrono::steady_clock::now();
    if (started == std::chrono::steady_clock::time_point{}) started = now;
    const float t = std::chrono::duration<float>(now - started).count();

    dl->AddRectFilled(ImVec2(0, 0), ImVec2(W, H), IM_COL32(0, 0, 0, 255));
    // Fundo: cobre a tela pela altura, recorte puxado pra esquerda (o Delsin fica a direita), zoom sem fim bem lento.
    if (art) {
        const float aw = float(art->Width), ah = float(art->Height);
        const float sc = std::max(H / ah, W / aw);
        const float vw = W / sc / aw, vh = H / sc / ah; // parte visivel, em UV
        const float u0 = (1.0f - vw) * 0.35f, v0 = (1.0f - vh) * 0.5f;
        const float zoom = 1.0f + 0.35f * (1.0f - std::exp(-t / 90.0f));
        const float cu = u0 + vw * 0.5f, cv = v0 + vh * 0.5f, hu = vw * 0.5f / zoom, hv = vh * 0.5f / zoom;
        dl->AddImage(art->GetTexRef(), ImVec2(0, 0), ImVec2(W, H), ImVec2(cu - hu, cv - hv), ImVec2(cu + hu, cv + hv),
                     Col(255, 255, 255, EaseOut(t / 0.9f)));
    }
    // Sombras da previa: embaixo e a esquerda.
    dl->AddRectFilledMultiColor(ImVec2(0, H * 0.45f), ImVec2(W, H), Col(5, 6, 8, 0), Col(5, 6, 8, 0),
                                Col(5, 6, 8, 0.84f), Col(5, 6, 8, 0.84f));
    dl->AddRectFilledMultiColor(ImVec2(0, 0), ImVec2(W * 0.55f, H), Col(5, 6, 8, 0.35f), Col(5, 6, 8, 0),
                                Col(5, 6, 8, 0), Col(5, 6, 8, 0.35f));

    const float L = 64.0f * s, base_y = 560.0f * s;
    if (logo) {
        const float lw = W * 0.40f, lh = lw * float(logo->Height) / float(logo->Width);
        const float ly = base_y + 36.0f * s - lh;
        // Sombra suave atras do logo (circulos sobrepostos: o ImGui nao tem gradiente radial).
        for (int i = 0; i < 10; ++i)
            dl->AddCircleFilled(ImVec2(L + lw * 0.45f, ly + lh * 0.38f), lw * (0.85f - i * 0.06f), Col(5, 6, 8, 0.06f), 48);
        const ImTextureRef tex = logo->GetTexRef();
        const auto strip = [&](float fy0, float fy1, float fx1) { // faixa do logo, em fracoes
            fx1 = std::clamp(fx1, 0.0f, 1.0f);
            if (fx1 <= 0.0f || fy1 <= fy0) return;
            dl->AddImage(tex, ImVec2(L, ly + fy0 * lh), ImVec2(L + fx1 * lw, ly + fy1 * lh), ImVec2(0, fy0),
                         ImVec2(fx1, fy1));
        };
        const float lines[3][2] = {{0.00f, 0.17f}, {0.17f, 0.50f}, {0.50f, 0.80f}};
        const float t0 = 0.7f, per = 0.42f;
        const int strips = 140;
        for (int i = 0; i < 3; ++i) {
            const float k = (t - t0 - i * per * 0.8f) / per;
            if (k <= 0.0f) continue;
            const float front = EaseOut(k) * (lw + 60.0f * s) - 30.0f * s;
            for (int j = int(lines[i][0] * strips); j < int(lines[i][1] * strips); ++j)
                strip(float(j) / strips, float(j + 1) / strips, (front + edge[j % edge.size()] * s) / lw);
            if (k < 1.0f)
                for (const auto& d : dots) {
                    const float dx = d[0] * lw;
                    if (d[1] < lines[i][0] || d[1] >= lines[i][1] || dx <= front || dx >= front + 70.0f * s) continue;
                    const float r = d[2] * s, cx = L + dx, cy = ly + d[1] * lh;
                    dl->AddImage(tex, ImVec2(cx - r, cy - r), ImVec2(cx + r, cy + r),
                                 ImVec2((dx - r) / lw, d[1] - r / lh), ImVec2((dx + r) / lw, d[1] + r / lh));
                }
        }
        // A tinta escorrendo, de cima pra baixo, depois do SON.
        const float kd = (t - t0 - 3 * per * 0.8f - per * 0.4f) / 0.7f;
        if (kd > 0.0f) strip(0.74f, 0.74f + 0.26f * EaseOut(kd), 1.0f);
    }

    // Titulo, barra e tempo: entram da esquerda depois do logo.
    const float k = EaseOut((t - (logo ? 2.6f : 0.6f)) / 0.55f);
    if (k <= 0.0f) return;
    ImFont* reg = DeckMenuImpl::font_regular ? DeckMenuImpl::font_regular : ImGui::GetFont();
    ImFont* bold = DeckMenuImpl::font_bold ? DeckMenuImpl::font_bold : reg;
    const float x = -640.0f * s + (L + 640.0f * s) * k;
    const int st = state.load();
    if (st != shown_state) { shown_state = st; state_started = now; }
    const u32 d = done.load(), n = total.load();
    const char* title = st == 2 ? "COMPILANDO SHADERS" : st == 1 ? "CARREGANDO SHADERS" : "PREPARANDO O JOGO";
    DeckMenuImpl::Text(dl, bold, 44.0f * s, ImVec2(x, base_y), DeckMenuImpl::Branco, title);
    if (n == 0) return;
    const float bw = 496.0f * s, by = base_y + 70.0f * s, frac = std::clamp(float(d) / float(n), 0.0f, 1.0f);
    dl->AddRectFilled(ImVec2(x, by), ImVec2(x + bw, by + 10.0f * s), IM_COL32(52, 52, 52, 255));
    dl->AddRectFilled(ImVec2(x, by), ImVec2(x + bw * frac, by + 10.0f * s), DeckMenuImpl::Amarelo);
    const auto dotted = [](u32 v) { // 20130 -> 20.130
        std::string t = std::to_string(v);
        for (int i = int(t.size()) - 3; i > 0; i -= 3) t.insert(size_t(i), ".");
        return t;
    };
    const std::string count = dotted(d) + " / " + dotted(n);
    char pct[16];
    std::snprintf(pct, sizeof(pct), "%u%%", unsigned(frac * 100.0f));
    DeckMenuImpl::Text(dl, reg, 22.0f * s, ImVec2(x, by + 24.0f * s), DeckMenuImpl::Texto, count.c_str());
    DeckMenuImpl::Text(dl, reg, 22.0f * s,
                       ImVec2(x + bw - DeckMenuImpl::Measure(reg, 22.0f * s, pct).x, by + 24.0f * s),
                       DeckMenuImpl::Amarelo, pct);
    const float elapsed = std::chrono::duration<float>(now - state_started).count();
    if (d > 0 && d < n && elapsed > 3.0f) {
        const unsigned left = unsigned(elapsed * float(n - d) / float(d)) + 1;
        char text[48];
        std::snprintf(text, sizeof(text), "Faltam cerca de %u:%02u", left / 60, left % 60);
        DeckMenuImpl::Text(dl, reg, 22.0f * s, ImVec2(x, by + 58.0f * s), DeckMenuImpl::Texto, text);
    }
}

} // namespace StartupImpl

// ---- steam-deck: caderno das sessoes de caca aos shaders ----
namespace CacaImpl {

std::mutex mutex;                    // o arquivo e as listas (present thread, compiladores)
std::FILE* file = nullptr;
bool opened = false;
std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
std::atomic<u32> compiles{0}, marks{0};
std::atomic<bool> mark_request{false};
bool l3_down = false, r3_down = false;
std::deque<std::pair<float, float>> frames;      // (t, ms) dos ultimos 10 s
std::deque<std::pair<float, std::string>> recent; // (t, "g 0x...") dos ultimos 10 s
float last_shot = -100.0f, flash_until = 0.0f;

float Now() {
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
}

bool Hunt() {
    static const bool on = [] {
        const char* p = std::getenv("TLG_MODO_CACA");
        return p && p[0] == '1';
    }();
    return on;
}

// mutex held
std::FILE* File() {
    if (!opened) {
        opened = true;
        if (const char* dir = std::getenv("TLG_CACA_DIR"); dir && *dir) {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            file = std::fopen((std::string(dir) + "/caderno.txt").c_str(), "a");
            if (file) {
                const std::time_t now = std::time(nullptr);
                char when[32];
                std::strftime(when, sizeof(when), "%F %T", std::localtime(&now));
                std::fprintf(file, "sessao %s%s\n", when, Hunt() ? " (modo caca)" : "");
                std::fflush(file);
            }
        }
    }
    return file;
}

void Shot(const char* why, float t) {
    static const bool overlays = [] { // TLG_SHOT_OVERLAY=1: o print leva a sobreposicao do port junto (testes)
        const char* p = std::getenv("TLG_SHOT_OVERLAY");
        return p && p[0] == '1';
    }();
    VideoCore::RequestScreenshot(overlays ? VideoCore::ScreenshotRequest::WithOverlays
                                          : VideoCore::ScreenshotRequest::GameOnly);
    last_shot = t;
    if (std::FILE* f = File()) {
        const std::time_t now = std::time(nullptr);
        char when[16];
        std::strftime(when, sizeof(when), "%H%M%S", std::localtime(&now));
        std::fprintf(f, "t=%.1f print (%s, arquivo de captura de ~%s)\n", t, why, when);
    }
}

void Compile(char kind, u64 hash) {
    std::scoped_lock lock{mutex};
    const float t = Now();
    ++compiles;
    flash_until = t + 1.2f;
    char text[40];
    std::snprintf(text, sizeof(text), "%c %#llx", kind, (unsigned long long)hash);
    recent.emplace_back(t, text);
    while (!recent.empty() && recent.front().first < t - 10.0f) recent.pop_front();
    if (std::FILE* f = File()) {
        std::fprintf(f, "t=%.1f compilou %s\n", t, text);
        std::fflush(f);
    }
}

long RssMb() {
    long pages = 0, rss = 0;
    if (std::FILE* f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
        std::fclose(f);
    }
    return rss * 4 / 1024;
}

// Present thread, cada quadro (fora da tela de abertura). Devolve o aviso a mostrar (vazio: nenhum).
std::string Frame(float ms) {
    if (!std::getenv("TLG_CACA_DIR")) return {};
    std::scoped_lock lock{mutex};
    const float t = Now();
    frames.emplace_back(t, ms);
    while (!frames.empty() && frames.front().first < t - 10.0f) frames.pop_front();
    // TLG_SHOT_EVERY_S=N (comparacao de imagem entre builds): um print a cada N s de jogo.
    static const float every = [] {
        const char* p = std::getenv("TLG_SHOT_EVERY_S");
        return p && *p ? float(std::atof(p)) : 0.0f;
    }();
    if (every > 0.0f && t - last_shot >= every) Shot("periodico", t);
    if (ms > 250.0f && ms < 5000.0f) {
        if (std::FILE* f = File()) std::fprintf(f, "t=%.1f quadro de %.0f ms\n", t, ms);
        if (t - last_shot > 20.0f) Shot("travada", t);
        if (file) std::fflush(file);
    }
    if (mark_request.exchange(false)) {
        const u32 n = ++marks;
        float worst = 0.0f;
        u32 slow = 0;
        for (const auto& [ft, fm] : frames) {
            worst = std::max(worst, fm);
            slow += fm > 50.0f;
        }
        const float span = frames.size() > 1 ? frames.back().first - frames.front().first : 0.0f;
        if (std::FILE* f = File()) {
            std::fprintf(f, "t=%.1f MOMENTO #%u: ultimos %.0f s %.1f FPS, pior %.0f ms, %u quadros acima de 50 ms, "
                            "memoria %ld MB, compilacoes:", t, n, span,
                         span > 0 ? float(frames.size()) / span : 0.0f, worst, slow, RssMb());
            for (const auto& [ct, c] : recent) std::fprintf(f, " [%.1f %s]", ct, c.c_str());
            std::fprintf(f, "\n");
        }
        Shot("momento", t);
        if (file) std::fflush(file);
        return "Momento #" + std::to_string(n) + " marcado";
    }
    return {};
}

// Modo caca: "SHADERS NOVOS: N" no canto superior direito, amarelo por um instante quando compila.
void DrawCounter() {
    if (!Hunt()) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float s = vp->Size.y / 800.0f;
    ImFont* reg = DeckMenuImpl::font_bold ? DeckMenuImpl::font_bold : ImGui::GetFont();
    char text[48];
    std::snprintf(text, sizeof(text), "SHADERS NOVOS: %u", compiles.load());
    const ImVec2 size = DeckMenuImpl::Measure(reg, 20.0f * s, text);
    const ImVec2 at(vp->Size.x - size.x - 16.0f * s, 12.0f * s);
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    dl->AddRectFilled(ImVec2(at.x - 8.0f * s, at.y - 4.0f * s), ImVec2(at.x + size.x + 8.0f * s, at.y + size.y + 4.0f * s),
                      IM_COL32(14, 15, 18, 170));
    const bool flash = Now() < flash_until;
    DeckMenuImpl::Text(dl, reg, 20.0f * s, at, flash ? DeckMenuImpl::Amarelo : DeckMenuImpl::Branco, text);
}

} // namespace CacaImpl

// ---- steam-deck: pular video (segurar A) ----
// No canto de baixo, no estilo do menu do port: "Carregando..." com um arco girando enquanto o jogo carrega por
// tras do video; depois o botao A com um anel amarelo que enche segurando e "PULAR" na fonte de titulo do jogo.
namespace SkipImpl {
std::atomic<int> state{0};
std::atomic<float> progress{0.0f};
std::chrono::steady_clock::time_point since{};
ImTextureData* art = nullptr; // a arte da tela de abertura (Delsin com o Deck) cobrindo a cena pulada
bool art_tried = false;

// imgui_mutex held, before NewFrame: carrega a arte ao pular e libera quando a cena volta.
void UpdateArt() {
    if (state.load() == 3) {
        if (art || art_tried) return;
        art_tried = true;
        TlgPng::Image image;
        if (const char* p = std::getenv("TLG_LOADING_ART"); p && *p && TlgPng::LoadRgba(p, image))
            art = StartupImpl::MakeTexture(image);
    } else {
        art_tried = false;
        if (!art) return;
        art->WantDestroyNextFrame = true;
        choice_retired.push_back(art);
        choice_retiring = true;
        art = nullptr;
    }
}

void Draw() {
    const int st = state.load();
    if (!st) return;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const float s = vp->Size.y / 800.0f;
    ImDrawList* dl = ImGui::GetForegroundDrawList();
    ImFont* reg = DeckMenuImpl::font_regular ? DeckMenuImpl::font_regular : ImGui::GetFont();
    ImFont* bold = DeckMenuImpl::font_bold ? DeckMenuImpl::font_bold : reg;
    const float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - since).count();
    const float r = 15.0f * s;
    const ImVec2 c(vp->Size.x - (st == 3 ? 330.0f : 150.0f) * s, vp->Size.y - 48.0f * s);
    if (st == 3) { // pulando: cobre a cena (o jogo segue com legendas e tela escura ate fechar o video)
        dl->AddRectFilled(ImVec2(0, 0), vp->Size, IM_COL32(0, 0, 0, 255));
        if (art) { // como na tela de abertura: cobre pela altura, recorte puxado pra esquerda, fade de 0,4 s
            const float W = vp->Size.x, H = vp->Size.y;
            const float aw = float(art->Width), ah = float(art->Height);
            const float sc = std::max(H / ah, W / aw);
            const float vw = W / sc / aw, vh = H / sc / ah;
            const float u0 = (1.0f - vw) * 0.35f, v0 = (1.0f - vh) * 0.5f;
            const float a = std::clamp(t / 0.4f, 0.0f, 1.0f);
            dl->AddImage(art->GetTexRef(), ImVec2(0, 0), ImVec2(W, H), ImVec2(u0, v0), ImVec2(u0 + vw, v0 + vh),
                         IM_COL32(255, 255, 255, int(a * 255.0f)));
            dl->AddRectFilledMultiColor(ImVec2(0, H * 0.45f), ImVec2(W, H), IM_COL32(5, 6, 8, 0), IM_COL32(5, 6, 8, 0),
                                        IM_COL32(5, 6, 8, 214), IM_COL32(5, 6, 8, 214));
        }
    }
    const char* text = st == 1 ? "CARREGANDO..." : st == 3 ? "PULANDO A CENA..." : "PULAR";
    const float size = 26.0f * s;
    const ImVec2 at(c.x + r + 12.0f * s, c.y - size * 0.55f);
    dl->AddText(bold, size, ImVec2(at.x + 1.5f * s, at.y + 1.5f * s), IM_COL32(0, 0, 0, 160), text);
    dl->AddText(bold, size, at, DeckMenuImpl::Branco, text);
    if (st == 1 || st == 3) {
        dl->PathArcTo(c, r, t * 6.0f, t * 6.0f + 4.4f, 24);
        dl->PathStroke(DeckMenuImpl::Amarelo, 0, 2.6f * s);
        return;
    }
    dl->AddCircleFilled(c, r, IM_COL32(14, 15, 18, 200), 32);
    dl->AddCircle(c, r, IM_COL32(233, 231, 225, 110), 32, 2.0f * s);
    const float p = std::clamp(progress.load(), 0.0f, 1.0f);
    if (p > 0.0f) {
        dl->PathArcTo(c, r, -1.5708f, -1.5708f + p * 6.2832f, 40);
        dl->PathStroke(DeckMenuImpl::Amarelo, 0, 3.0f * s);
    }
    // O botao de menu do Deck (tres tracos).
    for (int i = -1; i <= 1; ++i) {
        const float y = c.y + float(i) * 4.6f * s;
        dl->AddLine(ImVec2(c.x - 6.5f * s, y), ImVec2(c.x + 6.5f * s, y), DeckMenuImpl::Branco, 2.0f * s);
    }
}
} // namespace SkipImpl
#endif

void Menu() {
    auto& s = TlgSettings::Get();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(760.0f * base_scale, viewport->WorkSize.x * 0.95f),
                                    std::min(680.0f * base_scale, viewport->WorkSize.y * 0.9f)),
                             ImGuiCond_Appearing);
    bool keep_open = true;
    // tlg: the game's own title (param.sfo), not a fixed one.
    static const std::string heading = [] {
        const std::string_view title = Common::ElfInfo::Instance().Title();
        const char* app_name = std::getenv("TLG_APP_NAME");
        return std::string(!title.empty() ? title : app_name && *app_name ? app_name : "The Last Guardian") +
               (ViewMenu() ? std::string("  (View)")
                           : std::string(PadText("  (Insert / R3+L2)", "  (Insert / RS+LT)", "  (Insert / RS+ZL)")));
    }();
    if (!ImGui::Begin(heading.c_str(), &keep_open, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    footer_help = nullptr;
    constexpr int TabCount = 5;
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput) {
        const int step = ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) || ImGui::IsKeyPressed(ImGuiKey_E, false) ? 1
                         : ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) || ImGui::IsKeyPressed(ImGuiKey_Q, false)
                             ? -1
                             : 0;
        if (step) {
            requested_tab = (s.menu_tab + step + TabCount) % TabCount;
        }
    }
    const float footer = ImGui::GetFrameHeightWithSpacing() * 2.0f + ImGui::GetTextLineHeightWithSpacing() * 2.0f;
    if (ImGui::BeginTabBar("##tabs")) {
        static const char* const tabs[TabCount] = {"Overlay", TL("Image", "Imagem"), TL("Performance", "Desempenho"),
                                                   TL("Controls", "Controle"), "Extras"};
        for (int tab = 0; tab < TabCount; ++tab) {
            const ImGuiTabItemFlags flags = requested_tab == tab ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(tabs[tab], nullptr, flags)) {
                continue;
            }
            if (s.menu_tab != tab && requested_tab < 0) {
                s.menu_tab = tab;
                dirty = true;
            }
            if (ImGui::BeginChild("##body", ImVec2(0, -footer))) {
                switch (tab) {
                case 1: ImageTab(); break;
                case 2: PerformanceTab(); break;
                case 3: ControlsTab(); break;
                case 4: ExtrasTab(); break;
                default: OverlayTab(); break;
                }
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (requested_tab >= 0) {
            s.menu_tab = requested_tab;
            dirty = true;
            requested_tab = -1;
        }
        ImGui::EndTabBar();
    }

    ImGui::Separator();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", footer_help ? footer_help
                              : TL(PadText("L1 / R1 or Q / E: switch tabs. Settings are saved in tlg.ini.",
                                           "LB / RB or Q / E: switch tabs. Settings are saved in tlg.ini.",
                                           "L / R or Q / E: switch tabs. Settings are saved in tlg.ini."),
                                   PadText("L1 / R1 ou Q / E: trocar de aba. As configurações ficam salvas em tlg.ini.",
                                           "LB / RB ou Q / E: trocar de aba. As configurações ficam salvas em tlg.ini.",
                                           "L / R ou Q / E: trocar de aba. As configurações ficam salvas em tlg.ini.")));
    ImGui::PopTextWrapPos();
    if (const int changes = RestartChanges()) {
        ImGui::TextColored(ImVec4(0.85f, 0.72f, 0.45f, 1.0f), "%d change%s need%s a restart (unsaved progress is lost).",
                           changes, changes == 1 ? "" : "s", changes == 1 ? "s" : "");
        ImGui::SameLine();
        ImGui::BeginDisabled(!std::getenv("TLG_RESTART_COMMAND"));
        if (ImGui::Button(TL("Apply and restart", "Aplicar e reiniciar"))) {
            // The new launch confirms the options once the game shows; run.py restores the
            // previous ones if it never does (restart_unconfirmed).
            s.restart_unconfirmed = true;
            TlgSettings::Save();
            runtime_restart();
        }
        ImGui::EndDisabled();
    } else {
        ImGui::NewLine();
    }
    const float close_width = ImGui::CalcTextSize(TL("Close", "Fechar")).x + ImGui::GetStyle().FramePadding.x * 2;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - close_width);
    if (ImGui::Button(TL("Close", "Fechar"))) {
        keep_open = false;
    }
    ImGui::End();
    if (!keep_open) {
        SetOpen(false);
    }
}

// Enemy health bar from the damage hook's records (tlg_actors.cpp).
// Only the copied values are read, never the game's memory: an actor may be gone by now.
void HealthDisplay() {
    struct Seen {
        uint32_t seq = 0;
        std::chrono::steady_clock::time_point at{};
    };
    static Seen seen[2];
    const auto& s = TlgSettings::Get();
    if (!s.actor_watch_supported.load()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    TlgActorSlot slot[2];
    for (int i = 0; i < 2; ++i) {
        std::memcpy(&slot[i], const_cast<const TlgActorSlot*>(&tlg_actor_slots[i]), sizeof(slot[i]));
        if (slot[i].seq != seen[i].seq) {
            seen[i] = {slot[i].seq, now};
        }
    }
    const auto recent = [&](int i, int seconds) {
        return seen[i].seq != 0 && now - seen[i].at < std::chrono::seconds(seconds);
    };
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImDrawList* draw = ImGui::GetForegroundDrawList();
    const float scale = base_scale;
    if (s.enemy_health_bar && recent(1, 4) && TlgActors::Valid(slot[1].health, slot[1].max_health)) {
        // Minimal: a thin green bar centered on the screen's top edge; it shrinks toward the center
        // so it stays symmetric, with the health in numbers below.
        const ImVec2 display = ImGui::GetIO().DisplaySize;
        const float width = 260.0f * scale, height = 5.0f * scale, center = display.x * 0.5f;
        const float top = 36.0f * scale;
        const float fraction = TlgActors::Fraction(slot[1].health, slot[1].max_health);
        draw->AddRectFilled(ImVec2(center - width * 0.5f - scale, top - scale),
                            ImVec2(center + width * 0.5f + scale, top + height + scale),
                            IM_COL32(0, 0, 0, 140), 2 * scale);
        const float half = width * 0.5f * fraction;
        draw->AddRectFilled(ImVec2(center - half, top), ImVec2(center + half, top + height),
                            IM_COL32(70, 190, 80, 220), 2 * scale);
        char text[48];
        std::snprintf(text, sizeof(text), "%.0f / %.0f", std::ceil(slot[1].health), slot[1].max_health);
        const float font = ImGui::GetFontSize() * 0.9f;
        const ImVec2 size = ImGui::GetFont()->CalcTextSizeA(font, FLT_MAX, 0.0f, text);
        const ImVec2 at{center - size.x * 0.5f, top + height + 4.0f * scale};
        draw->AddText(ImGui::GetFont(), font, ImVec2(at.x + scale, at.y + scale), IM_COL32(0, 0, 0, 160), text);
        draw->AddText(ImGui::GetFont(), font, at, IM_COL32(225, 235, 220, 220), text);
    }
}

// The game's text dialog: what is typed, and how to finish (keyboard or controller).
void TextEntryBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.42f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSizeConstraints(ImVec2(viewport->WorkSize.x * 0.32f, 0.0f),
                                        ImVec2(viewport->WorkSize.x * 0.8f, FLT_MAX));
    ImGui::SetNextWindowBgAlpha(0.92f);
    ImGui::Begin("##text_entry", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextColored(ImVec4(0.85f, 0.72f, 0.45f, 1.0f), "%s", text_entry_prompt.c_str());
    ImGui::Separator();
    ImGui::SetWindowFontScale(1.4f);
    ImGui::Text("%s_", text_entry_text.c_str());
    ImGui::SetWindowFontScale(1.0f);
    ImGui::Separator();
    ImGui::TextDisabled("Keyboard: type, Backspace to erase, Enter = OK, Esc = cancel");
    ImGui::TextDisabled("%s", TL(PadText("Controller: Cross (A) = OK, Circle (B) = cancel", "Controller: A = OK, B = cancel", "Controller: B = OK, A = cancel"),
                                 PadText("Controle: Xis (A) = OK, Círculo (B) = cancelar", "Controle: A = OK, B = cancelar", "Controle: B = OK, A = cancelar")));
    ImGui::End();
}

// ---- Save dialogs: drawing. The layout is in design pixels of a 1280x720 screen, times k. ----
// steam-deck: as cores do menu do port (estilo do jogo): amarelo de destaque, branco gelo, cinzas neutros.
constexpr u32 kSaveAccent = 0xf2b631, kSaveText = 0xe9e7e1, kSaveMuted = 0xd9d6cf, kSaveDim = 0x8b8a86,
              kSavePanel = 0x0e0f12, kSaveLine = 0x2c2c2a, kSaveCard = 0x15161a, kSaveCardOn = 0x24231f,
              kSaveThumb = 0x1c1c1e, kSaveBorder = 0x6f6e6a, kSaveKnob = 0x4a4946, kSaveWarn = 0xf08a24,
              kSaveOk = 0xf2b631;
constexpr float kSaveRound = 0.0f; // cantos retos, como os paineis do menu

ImU32 SaveColor(u32 rgb, float alpha = 1.0f) {
    return IM_COL32(rgb >> 16, (rgb >> 8) & 255, rgb & 255, int(std::clamp(alpha, 0.0f, 1.0f) * 255.0f));
}

struct SavePainter {
    ImDrawList* dl;
    ImFont* font;
    float k;
    ImVec2 at; // the panel's top-left corner, in display pixels

    ImVec2 P(float x, float y) const { return ImVec2(at.x + x * k, at.y + y * k); }
    float Width(float size, const std::string& text) const {
        return font->CalcTextSizeA(size * k, FLT_MAX, 0.0f, text.c_str()).x / k;
    }
    // The text cut with "..." to fit `width` design pixels.
    std::string Fit(float size, const std::string& text, float width) const {
        if (Width(size, text) <= width) return text;
        std::string cut = text;
        while (!cut.empty()) {
            while (!cut.empty() && (u8(cut.back()) & 0xc0) == 0x80) cut.pop_back(); // UTF-8 tail
            if (!cut.empty()) cut.pop_back();
            if (Width(size, cut + "...") <= width) return cut + "...";
        }
        return "...";
    }
    // Text at (x, y) design pixels; bold is drawn twice, a hair apart (one embedded face).
    void Text(float x, float y, float size, u32 rgb, const std::string& text, bool bold = false,
              float alpha = 1.0f, float wrap = 0.0f) const {
        const ImU32 col = SaveColor(rgb, alpha);
        ImFont* face = bold && DeckMenuImpl::font_bold ? DeckMenuImpl::font_bold : font;
        dl->AddText(face, size * k * (bold && DeckMenuImpl::font_bold ? 1.12f : 1.0f), P(x, y), col, text.c_str(),
                    nullptr, wrap * k);
    }
    void TextRight(float right, float y, float size, u32 rgb, const std::string& text, bool bold = false) const {
        Text(right - Width(size, text), y, size, rgb, text, bold);
    }
    void Box(float x0, float y0, float x1, float y1, u32 fill, float rounding, float alpha = 1.0f) const {
        dl->AddRectFilled(P(x0, y0), P(x1, y1), SaveColor(fill, alpha), rounding * kSaveRound * k);
    }
    void Frame(float x0, float y0, float x1, float y1, u32 rgb, float rounding, float thickness,
               float alpha = 1.0f) const {
        dl->AddRect(P(x0, y0), P(x1, y1), SaveColor(rgb, alpha), rounding * kSaveRound * k, 0, thickness * k);
    }
    void Line(float x0, float y0, float x1, float y1, u32 rgb, float thickness, float alpha = 1.0f) const {
        dl->AddLine(P(x0, y0), P(x1, y1), SaveColor(rgb, alpha), thickness * k);
    }
    // A controller button: 0 confirm (bottom face button), 1 back (right one), centred at (x, y):
    // Cross / Circle, or the letters of TLG_GLYPHS=xbox (A / B) and nintendo (B / A).
    void Button(float x, float y, int which) const {
        const ImVec2 c = P(x, y);
        dl->AddCircleFilled(c, 11.0f * k, SaveColor(kSaveThumb));
        dl->AddCircle(c, 11.0f * k, SaveColor(kSaveKnob), 0, 1.0f * k);
        if (const char* letter = PadText(nullptr, which == 0 ? "A" : "B", which == 0 ? "B" : "A")) {
            const u32 rgb = which == 0 ? kSaveAccent : kSaveText;
            Text(x - Width(13.0f, letter) / 2, y - 8.0f, 13.0f, rgb, letter, true);
            return;
        }
        if (which == 0) {
            Line(x - 4, y - 4, x + 4, y + 4, 0x8fb6ff, 2.0f);
            Line(x + 4, y - 4, x - 4, y + 4, 0x8fb6ff, 2.0f);
        } else {
            dl->AddCircle(c, 5.0f * k, SaveColor(0xff8a8a), 0, 2.0f * k);
        }
    }
    // The panel itself: shadow, fill, border; then the discreet credit under its right corner.
    void Panel(float w, float h) const {
        dl->AddRectFilled(P(0, 18), P(w, h + 18), SaveColor(0x000000, 0.35f), 18.0f * kSaveRound * k);
        Box(0, 0, w, h, kSavePanel, 18.0f, 0.97f);
        Frame(0, 0, w, h, kSaveBorder, 18.0f, 1.0f);
        const std::string credit = "@hayatog-dev";
        Text(w - 6.0f - Width(11.0f, credit), h + 10.0f, 11.0f, 0x4a5568, credit);
    }
    // A save's picture (16:9) or a placeholder; `is_new` draws the dashed "+" slot.
    void Picture(float x, float y, float w, float h, ImTextureData* tex, bool is_new) const {
        if (tex) {
            dl->AddImageRounded(tex->GetTexRef(), P(x, y), P(x + w, y + h), ImVec2(0, 0), ImVec2(1, 1),
                                IM_COL32_WHITE, 10.0f * kSaveRound * k);
            Frame(x, y, x + w, y + h, 0x323d52, 10.0f, 1.0f);
            return;
        }
        if (is_new) {
            for (float t = 0; t < 1.0f; t += 0.1f) { // dashes around the slot
                Line(x + w * t, y, x + w * (t + 0.05f), y, kSaveKnob, 2.0f);
                Line(x + w * t, y + h, x + w * (t + 0.05f), y + h, kSaveKnob, 2.0f);
            }
            for (float t = 0; t < 1.0f; t += 0.18f) {
                Line(x, y + h * t, x, y + h * (t + 0.09f), kSaveKnob, 2.0f);
                Line(x + w, y + h * t, x + w, y + h * (t + 0.09f), kSaveKnob, 2.0f);
            }
            Line(x + w / 2, y + h / 2 - 11, x + w / 2, y + h / 2 + 11, kSaveDim, 2.0f);
            Line(x + w / 2 - 11, y + h / 2, x + w / 2 + 11, y + h / 2, kSaveDim, 2.0f);
            return;
        }
        Box(x, y, x + w, y + h, kSaveThumb, 10.0f);
        Frame(x, y, x + w, y + h, 0x323d52, 10.0f, 1.0f);
    }
};

SavePainter SavePainterFor(float w, float h) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    // 720p design scaled to the display, kept inside the screen on narrow windows.
    const float k = std::min(viewport->WorkSize.y / 720.0f, viewport->WorkSize.x * 0.94f / w);
    SavePainter p{ImGui::GetForegroundDrawList(),
                  DeckMenuImpl::font_regular ? DeckMenuImpl::font_regular : ImGui::GetFont(), k, {}};
    p.at = ImVec2(viewport->WorkPos.x + (viewport->WorkSize.x - w * k) * 0.5f,
                  viewport->WorkPos.y + (viewport->WorkSize.y - h * k) * 0.5f);
    return p;
}

void SaveBackdrop(float alpha) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::GetForegroundDrawList()->AddRectFilled(
        viewport->WorkPos, ImVec2(viewport->WorkPos.x + viewport->WorkSize.x, viewport->WorkPos.y + viewport->WorkSize.y),
        SaveColor(0x05070c, alpha));
}

std::string SaveWhen(s64 mtime) {
    if (mtime <= 0) return {};
    const time_t t = time_t(mtime);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char out[64];
    if (UiPortuguese()) {
        std::snprintf(out, sizeof(out), "%02d/%02d/%04d · %02d:%02d", tm.tm_mday, tm.tm_mon + 1,
                      tm.tm_year + 1900, tm.tm_hour, tm.tm_min);
    } else {
        static const char* months[] = {"Jan", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        const int hour = tm.tm_hour % 12 ? tm.tm_hour % 12 : 12;
        std::snprintf(out, sizeof(out), "%s %d, %d · %d:%02d %s", months[tm.tm_mon % 12], tm.tm_mday,
                      tm.tm_year + 1900, hour, tm.tm_min, tm.tm_hour < 12 ? "AM" : "PM");
    }
    return out;
}

std::string SaveAgo(s64 mtime) {
    const s64 seconds = s64(std::time(nullptr)) - mtime;
    if (mtime <= 0 || seconds < 0) return {};
    const bool pt = UiPortuguese();
    char out[64];
    if (seconds < 60) return pt ? "agora mesmo" : "just now";
    if (seconds < 3600) {
        const int m = int(seconds / 60);
        std::snprintf(out, sizeof(out), pt ? (m == 1 ? "há %d minuto" : "há %d minutos")
                                           : (m == 1 ? "%d minute ago" : "%d minutes ago"), m);
    } else if (seconds < 86400) {
        const int h = int(seconds / 3600);
        std::snprintf(out, sizeof(out), pt ? (h == 1 ? "há %d hora" : "há %d horas")
                                           : (h == 1 ? "%d hour ago" : "%d hours ago"), h);
    } else {
        const int d = int(seconds / 86400);
        if (d == 1) return pt ? "ontem" : "yesterday";
        std::snprintf(out, sizeof(out), pt ? "há %d dias" : "%d days ago", d);
    }
    return out;
}

std::string SaveSize(u64 bytes) {
    if (!bytes) return {};
    char out[32];
    if (bytes < 1024 * 1024) {
        std::snprintf(out, sizeof(out), "%llu KB", (unsigned long long)((bytes + 1023) / 1024));
    } else {
        std::snprintf(out, sizeof(out), "%.1f MB", double(bytes) / (1024.0 * 1024.0));
        if (UiPortuguese()) {
            for (char* c = out; *c; ++c) if (*c == '.') *c = ',';
        }
    }
    return out;
}

const char* SaveTitle(int kind) {
    return kind == 2 ? TL("Load game", "Carregar jogo") : kind == 3 ? TL("Delete save", "Excluir save")
                                                         : TL("Save game", "Salvar jogo");
}

ImTextureData* SaveTexture(size_t i) {
    return i < choice_textures.size() ? choice_textures[i] : nullptr;
}

// One save as a card: picture, the game's title and text, when it was saved.
void SaveCard(const SavePainter& p, float x0, float y0, float x1, const SaveRow& row, ImTextureData* tex,
              bool selected, float picture_w) {
    const float picture_h = picture_w * 9.0f / 16.0f, h = picture_h + 24.0f;
    if (selected) {
        p.Frame(x0 - 3, y0 - 3, x1 + 3, y0 + h + 3, kSaveAccent, 16.0f, 4.0f, 0.12f);
        p.Box(x0, y0, x1, y0 + h, kSaveCardOn, 14.0f);
        p.Frame(x0, y0, x1, y0 + h, kSaveAccent, 14.0f, 2.0f);
    } else {
        p.Box(x0, y0, x1, y0 + h, kSaveCard, 14.0f);
    }
    p.Picture(x0 + 16, y0 + 12, picture_w, picture_h, tex, row.is_new);
    const float text_x = x0 + 16 + picture_w + 18;
    std::string when, ago;
    if (row.is_new) {
        p.Text(text_x, y0 + h / 2 - 20, 17.0f, kSaveText, TL("New save", "Novo save"), true);
        p.Text(text_x, y0 + h / 2 + 4, 13.5f, kSaveMuted, TL("Create a new save", "Criar um save novo"));
        return;
    }
    when = SaveWhen(row.mtime);
    ago = SaveAgo(row.mtime);
    const std::string size = SaveSize(row.bytes);
    if (!size.empty()) ago = ago.empty() ? size : ago + " · " + size;
    const float right = x1 - 16;
    const float right_w = std::max(p.Width(14.0f, when), p.Width(12.5f, ago));
    const float text_w = right - text_x - (right_w > 0 ? right_w + 20 : 0);
    std::string line = row.subtitle;
    if (!row.detail.empty()) line = line.empty() ? row.detail : line + " · " + row.detail;
    for (char& c : line) if (c == '\n' || c == '\r') c = ' ';
    p.Text(text_x, y0 + h / 2 - 20, 17.0f, kSaveText, p.Fit(17.0f, row.title, text_w), true);
    p.Text(text_x, y0 + h / 2 + 4, 13.5f, kSaveMuted, p.Fit(13.5f, line, text_w));
    p.TextRight(right, y0 + h / 2 - 18, 14.0f, 0xdfe6f1, when, true);
    p.TextRight(right, y0 + h / 2 + 5, 12.5f, kSaveDim, ago);
}

void SaveListBox() {
    const int count = int(save_rows.size());
    const int shown = std::min(count, 4);
    const float w = 880, row_h = 96, gap = 10, header_h = 78, footer_h = 54, pad = 18;
    const float list_h = shown * row_h + std::max(shown - 1, 0) * gap + pad * 2;
    const float h = header_h + list_h + footer_h;
    SaveBackdrop(0.62f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);

    // Header: a floppy, the title, the game and the number of saves.
    const float ix = 28, iy = 25;
    p.Frame(ix, iy, ix + 26, iy + 26, kSaveAccent, 4.0f, 1.8f);
    p.Frame(ix + 6, iy, ix + 20, iy + 9, kSaveAccent, 1.0f, 1.8f);
    p.Frame(ix + 6, iy + 15, ix + 20, iy + 26, kSaveAccent, 1.0f, 1.8f);
    p.Text(70, 18, 24.0f, kSaveText, SaveTitle(save_kind), true);
    const int saves = count - int(std::count_if(save_rows.begin(), save_rows.end(), [](const SaveRow& r) { return r.is_new; }));
    char sub[160];
    const char* game = std::getenv("TLG_APP_NAME");
    std::snprintf(sub, sizeof(sub), "%s%s%d %s", game && *game ? game : "", game && *game ? " · " : "", saves,
                  saves == 1 ? "save" : "saves");
    p.Text(70, 48, 13.0f, kSaveDim, sub);
    p.Line(0, header_h, w, header_h, kSaveLine, 1.0f);

    // Rows, scrolled so the selected one is shown.
    if (choice_index < choice_first) choice_first = choice_index;
    if (choice_index >= choice_first + shown) choice_first = choice_index - shown + 1;
    choice_first = std::clamp(choice_first, 0, std::max(count - shown, 0));
    save_row_rects.assign(count, ImRect());
    for (int i = choice_first; i < choice_first + shown; ++i) {
        const float y = header_h + pad + (i - choice_first) * (row_h + gap);
        SaveCard(p, 20, y, w - 20, save_rows[i], SaveTexture(size_t(i)), i == choice_index, 128.0f);
        save_row_rects[i] = ImRect(p.P(20, y), p.P(w - 20, y + row_h));
    }
    if (count > shown) { // scroll bar
        const float track = list_h - pad * 2, y0 = header_h + pad;
        const float knob = track * shown / count, top = y0 + track * choice_first / count;
        p.Box(w - 11, y0, w - 8, y0 + track, kSaveLine, 2.0f);
        p.Box(w - 11, top, w - 8, top + knob, kSaveDim, 2.0f);
    }

    // Footer: what the buttons do.
    const float fy = header_h + list_h;
    p.Line(0, fy, w, fy, kSaveLine, 1.0f);
    const char* confirm = save_kind == 2 ? TL("Load", "Carregar") : save_kind == 3 ? TL("Delete", "Excluir")
                                                                                  : TL("Save here", "Salvar aqui");
    float x = 40;
    p.Button(x, fy + footer_h / 2, 0);
    p.Text(x + 18, fy + footer_h / 2 - 9, 13.5f, 0xb8c2d3, confirm);
    x += 18 + p.Width(13.5f, confirm) + 34;
    p.Button(x, fy + footer_h / 2, 1);
    p.Text(x + 18, fy + footer_h / 2 - 9, 13.5f, 0xb8c2d3, TL("Back", "Voltar"));
}

// The message's words for a system message type (sceSaveDataDialog), the game's own text or an
// error code. warn: drawn with the warning sign.
struct SaveMessageWords {
    std::string title, body;
    int buttons; // 0 OK, 1 yes/no, 3 OK/cancel
    bool warn;
};

SaveMessageWords SaveMessageFor(int kind, int code, u64 value, const std::string& text) {
    const bool pt = UiPortuguese();
    switch (code) {
    case 1:
        return {TL("No saved data", "Nenhum save encontrado"),
                TL("There is no saved data to load.", "Não há nenhum save para carregar."), 0, false};
    case 2:
        if (kind == 2) return {TL("Load this save?", "Carregar este save?"),
                               TL("Progress that was not saved will be lost.", "O progresso não salvo será perdido."), 1, false};
        if (kind == 3) return {TL("Delete this save?", "Excluir este save?"),
                               TL("This cannot be undone.", "Não dá para desfazer."), 1, true};
        return {TL("Save the game?", "Salvar o jogo?"), TL("A new save will be created.", "Um save novo será criado."), 1, false};
    case 3:
        return {TL("Overwrite this save?", "Sobrescrever este save?"),
                TL("The current progress will replace this save. This cannot be undone.",
                   "O progresso atual vai substituir este save. Não dá para desfazer."), 1, true};
    case 4:
    case 8:
        return {TL("Not enough space", "Sem espaço suficiente"),
                TL("Free some disk space and try again.", "Libere espaço no disco e tente de novo."), 0, true};
    case 6:
        return {TL("This save is damaged", "Este save está corrompido"),
                TL("It cannot be used.", "Não dá para usar este save."), 0, true};
    case 10:
        return {TL("A damaged save was found", "Um save corrompido foi encontrado"),
                TL("It was deleted.", "Ele foi excluído."), 0, true};
    case 11:
        return {TL("A damaged save was found", "Um save corrompido foi encontrado"),
                TL("A new save was created in its place.", "Um save novo foi criado no lugar."), 0, true};
    case 13:
        return {TL("A damaged save was found", "Um save corrompido foi encontrado"),
                TL("It was restored.", "Ele foi restaurado."), 0, true};
    case 14:
        return {TL("The saves are too large", "Os saves estão grandes demais"),
                TL("Delete some saves and try again.", "Exclua alguns saves e tente de novo."), 0, true};
    case 100:
        return {SaveTitle(kind), text, 0, false};
    case 101: {
        char body[96];
        std::snprintf(body, sizeof(body), pt ? "Código do erro: 0x%08X" : "Error code: 0x%08X", unsigned(value));
        return {TL("Save data error", "Erro nos dados salvos"), body, 0, true};
    }
    default:
        return {SaveTitle(kind), text, 0, false};
    }
}

void SaveMessageBox() {
    const SaveMessageWords words = SaveMessageFor(save_kind, message_code, message_value, message_text);
    const int buttons = message_buttons >= 0 ? message_buttons : words.buttons;
    const float w = 560, pad = 28, inner = w - pad * 2;
    // Height: title, the save's card, the wrapped text, the buttons.
    ImFont* font = ImGui::GetFont();
    const float k0 = std::min(ImGui::GetMainViewport()->WorkSize.y / 720.0f,
                              ImGui::GetMainViewport()->WorkSize.x * 0.94f / w);
    const float body_h = words.body.empty() ? 0.0f
        : font->CalcTextSizeA(14.5f * k0, FLT_MAX, inner * k0, words.body.c_str()).y / k0;
    const float card_h = message_has_entry ? 87.0f : 0.0f;
    float h = pad + 30 + 20 + (card_h ? card_h + 20 : 0) + (body_h ? body_h + 20 : 0) + 46 + pad;
    SaveBackdrop(0.72f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);

    float y = pad;
    if (words.warn) { // triangle
        const ImVec2 a = p.P(pad + 13, y + 2), b = p.P(pad + 26, y + 26), c = p.P(pad, y + 26);
        p.dl->AddTriangle(a, b, c, SaveColor(kSaveWarn), 2.0f * p.k);
        p.Line(pad + 13, y + 11, pad + 13, y + 18, kSaveWarn, 2.0f);
        p.dl->AddCircleFilled(p.P(pad + 13, y + 22), 1.4f * p.k, SaveColor(kSaveWarn));
    } else {
        p.dl->AddCircle(p.P(pad + 13, y + 14), 12.0f * p.k, SaveColor(kSaveAccent), 0, 2.0f * p.k);
        p.Text(pad + 8.5f, y + 4, 16.0f, kSaveAccent, "?", true);
    }
    p.Text(pad + 40, y + 1, 22.0f, kSaveText, p.Fit(22.0f, words.title, inner - 40), true);
    y += 30 + 20;
    if (message_has_entry) {
        p.Box(pad, y, w - pad, y + card_h, kSaveCard, 14.0f);
        p.Frame(pad, y, w - pad, y + card_h, kSaveBorder, 14.0f, 1.0f);
        p.Picture(pad + 14, y + 12, 112, 63, SaveTexture(0), message_entry.is_new);
        const float tx = pad + 14 + 112 + 16, tw = w - pad - 14 - tx;
        std::string line = message_entry.subtitle;
        if (!message_entry.detail.empty()) line = line.empty() ? message_entry.detail : line + " · " + message_entry.detail;
        for (char& c : line) if (c == '\n' || c == '\r') c = ' ';
        std::string when = SaveWhen(message_entry.mtime);
        const std::string ago = SaveAgo(message_entry.mtime);
        if (!when.empty() && !ago.empty()) when += " (" + ago + ")";
        p.Text(tx, y + 16, 16.0f, kSaveText,
               p.Fit(16.0f, message_entry.is_new ? std::string(TL("New save", "Novo save")) : message_entry.title, tw), true);
        p.Text(tx, y + 39, 13.0f, kSaveMuted, p.Fit(13.0f, line, tw));
        p.Text(tx, y + 59, 12.5f, kSaveDim, p.Fit(12.5f, when, tw));
        y += card_h + 20;
    }
    if (body_h) {
        p.Text(pad, y, 14.5f, 0xc3ccdb, words.body, false, 1.0f, inner);
        y += body_h + 20;
    }

    // Buttons, right-aligned: [no/cancel] [yes/OK]; the focused one is filled.
    const char* yes = buttons == 1 ? TL("Yes", "Sim") : "OK";
    const char* no = buttons == 1 ? TL("No", "Não") : TL("Cancel", "Cancelar");
    const bool two = buttons == 1 || buttons == 3;
    if (!two) message_focus = 1;
    float right = w - pad;
    save_button_rects[0] = save_button_rects[1] = ImRect();
    for (int b = 1; b >= (two ? 0 : 1); --b) {
        const char* label = b == 1 ? yes : no;
        const float bw = 22 + 20 + 10 + p.Width(15.0f, label) + 22, x0 = right - bw;
        const bool focused = b == message_focus;
        p.Box(x0, y, right, y + 46, focused ? kSaveAccent : kSaveCard, 12.0f);
        if (!focused) p.Frame(x0, y, right, y + 46, kSaveKnob, 12.0f, 1.0f);
        p.Button(x0 + 32, y + 23, b == 1 ? 0 : 1);
        p.Text(x0 + 52, y + 13, 15.0f, focused ? 0x0b0e14 : 0xdfe6f1, label, true);
        save_button_rects[b] = ImRect(p.P(x0, y), p.P(right, y + 46));
        right = x0 - 12;
    }
}

void SaveProgressBox(SaveClock::time_point now) {
    const float w = 520, h = 140;
    SaveBackdrop(0.45f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    // A spinning arc.
    const float t = std::chrono::duration<float>(now - progress_since).count();
    const ImVec2 c = p.P(28 + 13, 26 + 13);
    p.dl->PathArcTo(c, 11.0f * p.k, t * 6.0f, t * 6.0f + 4.4f, 24);
    p.dl->PathStroke(SaveColor(kSaveAccent), 0, 2.4f * p.k);
    const char* title = progress_kind == 2 ? TL("Loading...", "Carregando...")
                        : progress_kind == 3 ? TL("Deleting...", "Excluindo...")
                                             : TL("Saving...", "Salvando...");
    p.Text(66, 20, 21.0f, kSaveText, title, true);
    p.Text(66, 47, 13.0f, kSaveDim, TL("Do not close the game.", "Não feche o jogo."));
    const float bx0 = 28, bx1 = w - 28, by = 84;
    p.Box(bx0, by, bx1, by + 10, kSaveThumb, 5.0f);
    if (progress_percent >= 0) {
        const float f = std::clamp(progress_percent, 0, 100) / 100.0f;
        if (f > 0) p.Box(bx0, by, bx0 + (bx1 - bx0) * f, by + 10, kSaveAccent, 5.0f);
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%d%%", std::clamp(progress_percent, 0, 100));
        p.TextRight(bx1, by + 20, 13.0f, 0xdfe6f1, pct, true);
    } else { // no figure: a segment sweeps the bar
        const float span = (bx1 - bx0) * 0.3f, phase = std::fmod(t * 0.8f, 1.0f);
        const float s0 = bx0 + (bx1 - bx0 + span) * phase - span;
        p.Box(std::max(s0, bx0), by, std::min(s0 + span, bx1), by + 10, kSaveAccent, 5.0f);
    }
    if (!progress_text.empty()) {
        std::string text = progress_text;
        for (char& ch : text) if (ch == '\n' || ch == '\r') ch = ' ';
        p.Text(bx0, by + 20, 13.0f, kSaveMuted, p.Fit(13.0f, text, bx1 - bx0 - 60));
    }
}

void SaveToastBox(SaveClock::time_point now) {
    // steam-deck: discreet, as the Kewin asked: only a small "Salvando..." in the bottom-right
    // corner (no card, icon or game name), with a soft shadow so it reads on any scene.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const char* text = toast_kind == 2 ? TL("Loading...", "Carregando...")
                       : toast_kind == 3 ? TL("Deleting...", "Excluindo...")
                                         : TL("Saving...", "Salvando...");
    const float in = std::chrono::duration<float>(now - toast_since).count();
    const float out = std::chrono::duration<float>(toast_until - now).count();
    const float alpha = std::clamp(std::min(in / 0.2f, out / 0.5f), 0.0f, 1.0f) * 0.85f;
    const float k = viewport->WorkSize.y / 720.0f;
    SavePainter p{ImGui::GetForegroundDrawList(), ImGui::GetFont(), k, {}};
    const float w = p.Width(13.0f, text);
    p.at = ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - (22 + w) * k,
                  viewport->WorkPos.y + viewport->WorkSize.y - (22 + 16) * k);
    p.Text(1, 1, 13.0f, 0x000000, text, false, alpha * 0.6f);
    p.Text(0, 0, 13.0f, 0xe6e9ee, text, false, alpha);
}

// ---- The graffiti guide: how to paint without a gyro (sensor mode). Shown in the middle of the
// screen the first time sensor mode is used, closed with its button; the menu's Controls tab
// opens it again. imgui_mutex guards it. ----
std::atomic<bool> guide_active{false};
ImRect guide_button_rect;

void GuideBox() {
    const bool pt = UiPortuguese();
    const char* sticks = PadText("L3+R3", "LS+RS", "LS+RS");
    const char* shake = PadText(pt ? "✕ (Xis)" : "✕ (cross)", "A", "B");
    const char* spray = PadText("R2", "RT", "ZR");
    char steps[4][200];
    std::snprintf(steps[0], sizeof(steps[0]), pt ? "Toque em %s juntos: liga o modo sensor e a lata já fica de lado."
                                                 : "Tap %s together: sensor mode turns on and the can is held sideways.", sticks);
    std::snprintf(steps[1], sizeof(steps[1]), pt ? "Segure %s para sacudir a lata." : "Hold %s to shake the can.", shake);
    std::snprintf(steps[2], sizeof(steps[2]), pt ? "Segure %s para pintar e mire com o analógico esquerdo (pouco inclinado vai devagar)."
                                                 : "Hold %s to spray and aim with the left stick (a small push is slow).", spray);
    std::snprintf(steps[3], sizeof(steps[3]), pt ? "Toque em %s de novo para voltar a andar." : "Tap %s again to walk again.", sticks);
    const char* keys = TL("Keyboard: hold T or G (can sideways), Y shakes, F sprays, the arrow keys aim.",
                          "Teclado: segure T ou G (lata de lado), Y sacode, F pinta, as setas miram.");
    const char* again = TL("The Insert menu, Controls tab, shows this again.",
                           "O menu Insert, aba Controle, mostra isto de novo.");
    const float w = 640, pad = 30, inner = w - pad * 2, text_x = pad + 40;
    ImFont* font = ImGui::GetFont();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float k0 = std::min(viewport->WorkSize.y / 720.0f, viewport->WorkSize.x * 0.94f / w);
    auto height = [&](const char* text, float size, float width) {
        return font->CalcTextSizeA(size * k0, FLT_MAX, width * k0, text).y / k0;
    };
    float step_h[4], h = pad + 30 + 8 + 20 + 22;
    for (int i = 0; i < 4; ++i) {
        step_h[i] = std::max(28.0f, height(steps[i], 15.0f, inner - 40));
        h += step_h[i] + 12;
    }
    const float keys_h = height(keys, 13.0f, inner);
    h += 8 + keys_h + 22 + 46 + pad;
    SaveBackdrop(0.70f);
    SavePainter p = SavePainterFor(w, h);
    p.Panel(w, h);
    float y = pad;
    p.Text(pad, y, 23.0f, kSaveText, TL("Graffiti without a gyro", "Grafite sem giroscópio"), true);
    y += 30 + 8;
    p.Text(pad, y, 14.0f, kSaveMuted, TL("No motion sensor on your controller? Use sensor mode:",
                                         "Seu controle não tem sensor de movimento? Use o modo sensor:"));
    y += 20 + 22;
    for (int i = 0; i < 4; ++i) {
        const ImVec2 c = p.P(pad + 13, y + 11);
        p.dl->AddCircleFilled(c, 13.0f * p.k, SaveColor(kSaveCardOn));
        p.dl->AddCircle(c, 13.0f * p.k, SaveColor(kSaveAccent), 0, 1.5f * p.k);
        char number[4];
        std::snprintf(number, sizeof(number), "%d", i + 1);
        p.Text(pad + 13 - p.Width(14.0f, number) / 2, y + 2, 14.0f, kSaveAccent, number, true);
        p.Text(text_x, y + 1, 15.0f, kSaveText, steps[i], false, 1.0f, inner - 40);
        y += step_h[i] + 12;
    }
    y += 8;
    p.Text(pad, y, 13.0f, kSaveDim, keys, false, 1.0f, inner);
    y += keys_h + 22;
    p.Text(pad, y + 15, 12.5f, 0x6f7c92, again, false, 1.0f, inner - 180);
    const char* ok = TL("Got it", "Entendi");
    const float bw = 22 + 20 + 10 + p.Width(15.0f, ok) + 22, x0 = w - pad - bw;
    p.Box(x0, y, w - pad, y + 46, kSaveAccent, 12.0f);
    p.Button(x0 + 32, y + 23, 0);
    p.Text(x0 + 52, y + 13, 15.0f, 0x0b0e14, ok, true);
    guide_button_rect = ImRect(p.P(x0, y), p.P(w - pad, y + 46));
}

// imgui_mutex held.
void OpenGuide() {
    guide_button_rect = ImRect();
    guide_active = true;
    ImGui::GetIO().MouseDrawCursor = true;
}

void CloseGuide() {
    guide_active = false;
    ImGui::GetIO().MouseDrawCursor = menu_open;
    auto& s = TlgSettings::Get();
    if (!s.graffiti_guide_seen.exchange(true)) {
        TlgSettings::Save();
    }
}

// Present thread, imgui_mutex held, after NewFrame: whichever save dialog is up.
void SaveDialogs() {
    const auto now = SaveClock::now();
    if (guide_active) {
        GuideBox();
        return;
    }
    if (choice_active) {
        SaveListBox();
    } else if (message_active) {
        SaveMessageBox();
    } else if (progress_on || now < progress_until) {
        SaveProgressBox(now);
    }
    if (now < toast_until && !choice_active && !message_active) {
        SaveToastBox(now);
    }
}

void LoadingScreen() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x * 0.4f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f);
    ImGui::Begin("##loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                     ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing);
    const u32 done = loading_done, total = loading_total;
    const bool compile = loading_compile;
    ImGui::SetWindowFontScale(1.6f);
    ImGui::TextUnformatted(compile ? TL("Compiling shaders...", "Compilando shaders...")
                           : total ? TL("Loading shader cache...", "Carregando o cache de shaders...")
                                   : TL("Preparing graphics...", "Preparando os gráficos..."));
    ImGui::SetWindowFontScale(1.0f);
    if (total) {
        char count[64];
        std::snprintf(count, sizeof(count), "%u / %u (%u%%)", done, total,
                      u32(u64(done) * 100 / total));
        ImGui::ProgressBar(float(done) / float(total), ImVec2(-1.0f, 0.0f), count);
    }
    if (compile && total && done > 0 && done < total) {
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                             loading_compile_started)
                                   .count();
        const u32 left = u32(elapsed * double(total - done) / double(done)) + 1;
        ImGui::Text(TL("About %u:%02u left", "Faltam cerca de %u:%02u"), left / 60, left % 60);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", compile ? TL("Only on the first start on this computer and after updates: "
                                           "then the game runs without compiling shaders.",
                                           "Só na primeira abertura neste computador e depois de "
                                           "atualizações: depois o jogo roda sem compilar shaders.")
                                      : TL("The first start after an update takes longer.",
                                           "A primeira abertura depois de uma atualização demora mais."));
    ImGui::PopTextWrapPos();
    ImGui::End();
}

void FpsCounter() {
    const auto& s = TlgSettings::Get();
    const std::string text = OverlayText();
    if (text.empty()) {
        return;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    const int corner = s.overlay_corner;
    const bool right = corner == 1 || corner == 3, bottom = corner >= 2;
    ImGui::SetNextWindowPos(ImVec2(right ? viewport->WorkPos.x + viewport->WorkSize.x - pad : viewport->WorkPos.x + pad,
                                   bottom ? viewport->WorkPos.y + viewport->WorkSize.y - pad : viewport->WorkPos.y + pad),
                            ImGuiCond_Always, ImVec2(right ? 1.0f : 0.0f, bottom ? 1.0f : 0.0f));
    ImGui::SetNextWindowBgAlpha(s.overlay_opacity / 100.0f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::SetWindowFontScale(s.overlay_scale / 100.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::End();
}

std::string notice_text;
std::chrono::steady_clock::time_point notice_until;

void NoticeBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + 40.0f * base_scale),
                            ImGuiCond_Always, ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.75f);
    ImGui::Begin("##notice", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::SetWindowFontScale(1.3f);
    ImGui::TextUnformatted(notice_text.c_str());
    ImGui::End();
}

// The mouse's touchpad (touchpad mode): the pad as the game sees it and the finger on it.
std::atomic<bool> mouse_pad_on{false}, mouse_pad_touch{false}, mouse_pad_click{false};
std::atomic<float> mouse_pad_x{0.5f}, mouse_pad_y{0.5f};

void MousePadBox() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float w = 240.0f * base_scale, h = w * 942.0f / 1919.0f;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 24.0f * base_scale,
                                   viewport->WorkPos.y + viewport->WorkSize.y - 24.0f * base_scale),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.45f);
    ImGui::SetNextWindowSize(ImVec2(w + 16.0f * base_scale, h + 16.0f * base_scale));
    ImGui::Begin("##mousepad", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoSavedSettings);
    const ImVec2 origin = ImVec2(ImGui::GetWindowPos().x + 8.0f * base_scale, ImGui::GetWindowPos().y + 8.0f * base_scale);
    ImDrawList* list = ImGui::GetWindowDrawList();
    const bool click = mouse_pad_click, touch = mouse_pad_touch;
    list->AddRectFilled(origin, ImVec2(origin.x + w, origin.y + h), IM_COL32(20, 20, 20, 160), 6.0f * base_scale);
    list->AddRect(origin, ImVec2(origin.x + w, origin.y + h), click ? IM_COL32(255, 200, 60, 255) : IM_COL32(220, 220, 220, 200),
                  6.0f * base_scale, 0, 2.0f * base_scale);
    const ImVec2 finger(origin.x + std::clamp(mouse_pad_x.load(), 0.0f, 1.0f) * w,
                        origin.y + std::clamp(mouse_pad_y.load(), 0.0f, 1.0f) * h);
    const float r = 7.0f * base_scale;
    if (touch) {
        list->AddCircleFilled(finger, r, click ? IM_COL32(255, 200, 60, 255) : IM_COL32(80, 200, 255, 255));
    } else {
        list->AddCircle(finger, r, IM_COL32(255, 255, 255, 220), 0, 2.0f * base_scale);
    }
    ImGui::End();
}

void BackgroundLoading() {
    const u32 done = background_done, total = std::max(background_total.load(), 1u);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + pad,
                                   viewport->WorkPos.y + viewport->WorkSize.y - pad),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.35f);
    ImGui::Begin("##background_loading", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextDisabled("Loading shader cache %u%%", u32(u64(done) * 100 / total));
    ImGui::End();
}

} // namespace

void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count) {
    std::scoped_lock lock{imgui_mutex};
    if (initialized) {
        return;
    }
    IMGUI_CHECKVERSION();
    ImGuiContext* const previous_context = ImGui::GetCurrentContext();
    overlay_context = ImGui::CreateContext();
    ImGui::SetCurrentContext(overlay_context);
#ifdef TLG_RENDERER_V2
    // The renderer's own context stays current for its presenter.
    struct Restore { ImGuiContext* c; ~Restore() { ImGui::SetCurrentContext(c); } } restore{previous_context};
#else
    (void)previous_context;
#endif
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // window positions are not kept
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.BackendPlatformName = "tlg";

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(tlg_font_ttf),
                                   int(tlg_font_ttf_end - tlg_font_ttf), 18.0f, &font_config);
    DeckMenuImpl::LoadFonts(io);

    const vk::Instance vk_instance = instance.GetInstance();
    ImGui_ImplVulkan_LoadFunctions(
        instance.ApiVersion(),
        [](const char* name, void* user) {
            return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
                *static_cast<const vk::Instance*>(user), name);
        },
        const_cast<vk::Instance*>(&vk_instance));

    const VkFormat color_format = static_cast<VkFormat>(format);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = instance.ApiVersion();
    info.Instance = vk_instance;
    info.PhysicalDevice = instance.GetPhysicalDevice();
    info.Device = instance.GetDevice();
    info.QueueFamily = instance.GetGraphicsQueueFamilyIndex();
    info.Queue = instance.GetGraphicsQueue();
    // Font atlas plus the save list pictures (up to 33, and a closed list's until freed).
    info.DescriptorPoolSize = 128;
    info.MinImageCount = std::max(image_count, 2u);
    info.ImageCount = std::max(image_count, 2u);
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    if (!ImGui_ImplVulkan_Init(&info)) {
        std::printf("Overlay: ImGui Vulkan backend init failed\n");
        ImGui::DestroyContext();
        return;
    }
    overlay_instance = &instance;
    initialized = true;
    std::printf("Overlay: menu ready (Insert or R3+L2)\n");
}

void UpdateTextInput(SDL_Window* window) {
    if (!overlay_context) return; // before Init
    bool want = false;
    {
        std::scoped_lock lock{imgui_mutex};
        want = initialized && menu_open && overlay_context->IO.WantTextInput;
    }
    if (want != SDL_TextInputActive(window)) {
        if (want) {
            SDL_StartTextInput(window);
        } else {
            SDL_StopTextInput(window);
        }
    }
}

// imgui_mutex held. The backend frees a texture some frames after its last use.
void RetireChoiceTextures() {
    for (ImTextureData* tex : choice_textures) {
        if (tex) {
            tex->WantDestroyNextFrame = true;
            choice_retired.push_back(tex);
        }
    }
    choice_textures.clear();
    choice_pictures.clear();
    choice_retiring = !choice_retired.empty();
}

// imgui_mutex held, before NewFrame: new pictures become textures, destroyed ones are freed.
void UpdateChoiceTextures() {
    if (!choice_pictures.empty()) {
        choice_textures.assign(choice_pictures.size(), nullptr);
        for (size_t i = 0; i < choice_pictures.size(); ++i) {
            const TlgPng::Image& picture = choice_pictures[i];
            if (picture.rgba.empty()) {
                continue;
            }
            ImTextureData* tex = IM_NEW(ImTextureData)();
            tex->Create(ImTextureFormat_RGBA32, picture.width, picture.height);
            std::memcpy(tex->GetPixels(), picture.rgba.data(), picture.rgba.size());
            ImGui::RegisterUserTexture(tex);
            choice_textures[i] = tex;
        }
        choice_pictures.clear();
    }
    std::erase_if(choice_retired, [](ImTextureData* tex) {
        if (tex->Status != ImTextureStatus_Destroyed) {
            return false;
        }
        ImGui::UnregisterUserTexture(tex);
        IM_DELETE(tex);
        return true;
    });
    choice_retiring = !choice_retired.empty();
}

// The guide takes the buttons, keys and mouse until it is closed (its button, cross/A, Enter,
// Esc, circle/B); the game's input is held neutral meanwhile.
bool HandleGuideEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    bool close = false;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN: {
        const SDL_Keycode key = event.key.key;
        close = key == SDLK_RETURN || key == SDLK_KP_ENTER || key == SDLK_SPACE || key == SDLK_ESCAPE;
        break;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        close = event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH || event.gbutton.button == SDL_GAMEPAD_BUTTON_EAST;
        break;
    case SDL_EVENT_MOUSE_MOTION: {
        const float density = PixelDensity(event.motion.windowID);
        overlay_context->IO.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const ImVec2 at = overlay_context->IO.MousePos;
        close = event.button.button == SDL_BUTTON_LEFT && guide_button_rect.GetWidth() > 0 &&
                guide_button_rect.Contains(at);
        break;
    }
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_WHEEL:
        return true;
    default:
        return false;
    }
    if (close) {
        CloseGuide();
    }
    return true;
}

void FinishSaveDialog() {
    RetireChoiceTextures();
    ImGui::GetIO().MouseDrawCursor = menu_open;
}

// The save list and messages take the keys, buttons and mouse that answer them; the rest stays
// with the game (held neutral by CapturesInput).
bool HandleSaveEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    int move = 0, finish = 0; // finish: 1 accept, 2 back
    bool moved = false, click = false;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN: {
        const SDL_Keycode key = event.key.key;
        move = key == SDLK_UP || key == SDLK_W || key == SDLK_LEFT || key == SDLK_A ? -1
               : key == SDLK_DOWN || key == SDLK_S || key == SDLK_RIGHT || key == SDLK_D ? 1
                                                                                      : 0;
        finish = key == SDLK_RETURN || key == SDLK_KP_ENTER || key == SDLK_SPACE ? 1
                 : key == SDLK_ESCAPE || key == SDLK_BACKSPACE                   ? 2
                                                                                 : 0;
        break;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN: {
        const u8 button = event.gbutton.button;
        move = button == SDL_GAMEPAD_BUTTON_DPAD_UP || button == SDL_GAMEPAD_BUTTON_DPAD_LEFT ? -1
               : button == SDL_GAMEPAD_BUTTON_DPAD_DOWN || button == SDL_GAMEPAD_BUTTON_DPAD_RIGHT ? 1
                                                                                                  : 0;
        finish = button == SDL_GAMEPAD_BUTTON_SOUTH ? 1 : button == SDL_GAMEPAD_BUTTON_EAST ? 2 : 0;
        break;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        const float density = PixelDensity(event.motion.windowID);
        save_mouse = ImVec2(event.motion.x * density, event.motion.y * density);
        overlay_context->IO.AddMousePosEvent(save_mouse.x, save_mouse.y);
        moved = true;
        break;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        click = event.button.button == SDL_BUTTON_LEFT;
        finish = event.button.button == SDL_BUTTON_RIGHT ? 2 : 0;
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        move = event.wheel.y > 0 ? -1 : event.wheel.y < 0 ? 1 : 0;
        break;
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        return true;
    default:
        return false;
    }
    if (message_active) {
        const bool two = (message_buttons >= 0 ? message_buttons
                          : SaveMessageFor(save_kind, message_code, message_value, message_text).buttons) % 2 == 1;
        if (two && move) message_focus = move < 0 ? 0 : 1;
        for (int b = 0; b < 2 && (moved || click); ++b) {
            if (save_button_rects[b].GetWidth() > 0 && save_button_rects[b].Contains(save_mouse)) {
                message_focus = b;
                finish = click ? 1 : finish;
            }
        }
        if (finish) {
            message_result = !two || (finish == 1 && message_focus == 1) ? 1 : 2;
            message_active = false;
            FinishSaveDialog();
        }
        return true;
    }
    const int count = int(save_rows.size());
    if (move && count) {
        choice_index = (choice_index + move + count) % count;
    }
    for (int i = 0; i < int(save_row_rects.size()) && (moved || click); ++i) {
        if (save_row_rects[i].GetWidth() > 0 && save_row_rects[i].Contains(save_mouse)) {
            choice_index = i;
            finish = click ? 1 : finish;
        }
    }
    if (finish) {
        choice_result = finish == 1 && count ? choice_index : -2;
        choice_active = false;
        FinishSaveDialog();
    }
    return true;
}

bool HandleEvent(const SDL_Event& event) {
    if (!overlay_context) return false; // before Init
    std::scoped_lock lock{imgui_mutex};
    if (!initialized) {
        return false;
    }
    if (guide_active && !menu_open) {
        return HandleGuideEvent(event);
    }
    if ((choice_active || message_active) && !menu_open) {
        return HandleSaveEvent(event);
    }
    ImGuiIO& io = overlay_context->IO; // not made current: other threads draw with ImGui
    const bool is_open = menu_open;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = event.type == SDL_EVENT_KEY_DOWN;
        static const bool capture_key = [] {
            const char* env = std::getenv("TLG_FRAME_CAPTURE_KEY");
            return env && *env == '1';
        }();
        if (down && !event.key.repeat && capture_key && event.key.key == SDLK_F11) {
#ifndef TLG_RENDERER_V2
            Vulkan::FrameCapture::Request();
#endif
            return true;
        }
        // tlg: F10 shows or hides the performance overlay (FPS counter) at any time.
        if (down && !event.key.repeat && event.key.key == SDLK_F10) {
            auto& settings = TlgSettings::Get();
            settings.show_fps = !settings.show_fps;
            return true;
        }
        if (down && !event.key.repeat &&
            (event.key.key == SDLK_INSERT || (is_open && event.key.key == SDLK_ESCAPE))) {
            SetOpen(event.key.key == SDLK_INSERT ? !is_open : false);
            return true;
        }
        if (!is_open) {
            return false;
        }
        io.AddKeyEvent(ImGuiMod_Ctrl, (event.key.mod & SDL_KMOD_CTRL) != 0);
        io.AddKeyEvent(ImGuiMod_Shift, (event.key.mod & SDL_KMOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiMod_Alt, (event.key.mod & SDL_KMOD_ALT) != 0);
        if (const ImGuiKey key = KeyFromSdl(event.key.key); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        const bool down = event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        const u8 button = event.gbutton.button;
        // steam-deck: View closes the open menu (runtime_pad opens it, after pausing the game).
        if (button == SDL_GAMEPAD_BUTTON_BACK && is_open) {
            if (down) {
                const char* classic_env = std::getenv("TLG_CLASSIC_MENU");
                if (DeckMenuImpl::classic || (classic_env && classic_env[0] == '1')) {
                    SetOpen(false);
                } else {
                    DeckMenuImpl::close_request = true;
                }
            }
            return true;
        }
#ifdef TLG_RENDERER_V2
        // steam-deck: L3+R3 juntos marcam o momento (caderno da sessao, TLG_CACA_DIR). O jogo recebe os botoes normal.
        if (button == SDL_GAMEPAD_BUTTON_LEFT_STICK || button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
            bool& held = button == SDL_GAMEPAD_BUTTON_LEFT_STICK ? CacaImpl::l3_down : CacaImpl::r3_down;
            const bool was_both = CacaImpl::l3_down && CacaImpl::r3_down;
            held = down;
            if (!was_both && CacaImpl::l3_down && CacaImpl::r3_down && std::getenv("TLG_CACA_DIR"))
                CacaImpl::mark_request = true;
        }
#endif
        // R3+L2 opens the menu: L3+R3 is a game action in God of War III. steam-deck: View only.
        if (button == SDL_GAMEPAD_BUTTON_RIGHT_STICK && !ViewMenu()) {
            r3_down = down;
            if (down && l2_down) {
                SetOpen(!is_open);
                return true;
            }
        }
        if (!is_open) {
            return false;
        }
        if (const ImGuiKey key = KeyFromGamepad(button); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        if (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) {
            const bool down = event.gaxis.value > 16000;
            if (down && !l2_down && r3_down && !ViewMenu()) {
                l2_down = true;
                SetOpen(!is_open);
                return true;
            }
            l2_down = down;
        }
        return false;
    }
    case SDL_EVENT_TEXT_INPUT: {
        // Typed characters (Ctrl+click on a slider, a text field): key events alone erase but
        // do not type. SDL sends them while text input is on (UpdateTextInput).
        if (!is_open) {
            return false;
        }
        io.AddInputCharactersUTF8(event.text.text);
        return true;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        if (!is_open) {
            return false;
        }
        const float density = PixelDensity(event.motion.windowID);
        io.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (!is_open) {
            return false;
        }
        const int button = event.button.button == SDL_BUTTON_LEFT    ? 0
                           : event.button.button == SDL_BUTTON_RIGHT  ? 1
                           : event.button.button == SDL_BUTTON_MIDDLE ? 2
                                                                      : -1;
        if (button >= 0) {
            io.AddMouseButtonEvent(button, event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        if (!is_open) {
            return false;
        }
        io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);
        return true;
    default:
        return false;
    }
}

// After "Apply and restart", the new options are kept once the game has shown about ten
// seconds of frames after loading (run.py restores the previous ones otherwise).
void ConfirmRestart() {
    static u32 frames = 0;
    auto& s = TlgSettings::Get();
    if (!s.restart_unconfirmed || loading || ++frames < 600) {
        return;
    }
    s.restart_unconfirmed = false;
    TlgSettings::Save();
    std::puts("Settings: graphics options confirmed after restart");
}

void SetBackgroundLoading(u32 done, u32 total) {
    background_done = done;
    background_total = total;
}

#ifdef TLG_RENDERER_V2
void SetStartup(bool on) {
    StartupImpl::active = on;
}
void SetStartupProgress(int state, u32 done, u32 total) {
    StartupImpl::state = state;
    StartupImpl::done = done;
    StartupImpl::total = total;
}
bool StartupActive() {
    return StartupImpl::active;
}
void NoteCompile(char kind, u64 hash) {
    CacaImpl::Compile(kind, hash);
}
void MovieSkip(int state, float progress) {
    if (state && !SkipImpl::state.load()) SkipImpl::since = std::chrono::steady_clock::now();
    SkipImpl::progress = progress;
    SkipImpl::state = state;
}
#endif

void SetLoadingCompile(bool compile) {
    if (compile && !loading_compile) {
        loading_compile_started = std::chrono::steady_clock::now();
    }
    loading_compile = compile;
}

void SetLoading(bool active, u32 done, u32 total) {
    loading_done = done;
    loading_total = total;
    loading = active;
}

void Notice(const std::string& text, float seconds) {
    std::scoped_lock lock{imgui_mutex};
    notice_text = text;
    notice_until = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(static_cast<int>(seconds * 1000.0f));
}

static bool NoticeActive() {
    return std::chrono::steady_clock::now() < notice_until;
}

void SetMousePad(bool active, float x, float y, bool touching, bool clicking) {
    mouse_pad_x = x;
    mouse_pad_y = y;
    mouse_pad_touch = touching;
    mouse_pad_click = clicking;
    mouse_pad_on = active;
}

bool Visible() {
    const auto& s = TlgSettings::Get();
    const auto now = std::chrono::steady_clock::now();
    static const char* steam_file = std::getenv("TLG_STEAM_PAUSE_FILE");
    static auto steam_checked = std::chrono::steady_clock::time_point{};
    static bool steam_file_present = false;
    if (steam_file && *steam_file && now - steam_checked > std::chrono::milliseconds(50)) {
        steam_checked = now;
        std::error_code ec;
        steam_file_present = std::filesystem::exists(steam_file, ec);
    }
#ifdef TLG_RENDERER_V2
    if (initialized && (StartupImpl::active || CacaImpl::Hunt() || SkipImpl::state.load())) return true;
#endif
    return initialized && (loading || background_total || menu_open || open_request.load() >= 0 ||
                           steam_file_present || DeckMenuImpl::steam_paused || text_entry_active || choice_active ||
                           message_active || guide_active || progress_on || now < progress_until || now < toast_until ||
                           NoticeActive() || mouse_pad_on ||
                           choice_retiring || s.show_fps ||
                           (s.actor_watch_supported && s.enemy_health_bar));
}

bool CapturesInput() {
    return menu_open || text_entry_active || choice_active || message_active || guide_active;
}

bool MenuOpen() {
    return menu_open;
}

void RequestOpen(bool open) {
    open_request.store(open ? 1 : 0, std::memory_order_release);
}

void ShowGraffitiGuide(bool force) {
    std::scoped_lock lock{imgui_mutex};
    if (!force && TlgSettings::Get().graffiti_guide_seen) {
        return;
    }
    OpenGuide();
}

void BeginSaveList(int kind, std::vector<SaveRow> rows, std::vector<TlgPng::Image> pictures, int focus) {
    std::scoped_lock lock{imgui_mutex};
    RetireChoiceTextures();
    choice_pictures = std::move(pictures);
    choice_pictures.resize(rows.size());
    save_kind = kind;
    save_rows = std::move(rows);
    save_row_rects.clear();
    choice_index = focus >= 0 && focus < int(save_rows.size()) ? focus : 0;
    choice_first = 0;
    choice_result = -1;
    message_active = false;
    progress_on = false;
    choice_active = true;
    overlay_context->IO.MouseDrawCursor = true;
}

int PollSaveList() {
    return choice_result;
}

void BeginSaveMessage(int kind, int code, u64 value, const std::string& text, int buttons,
                      const SaveRow* entry, TlgPng::Image picture) {
    std::scoped_lock lock{imgui_mutex};
    RetireChoiceTextures();
    message_has_entry = entry != nullptr;
    if (entry) {
        message_entry = *entry;
        choice_pictures.clear();
        choice_pictures.push_back(std::move(picture));
    }
    save_kind = kind;
    message_code = code;
    message_value = value;
    message_text = text;
    message_buttons = buttons;
    // Deleting starts on "No"; everything else on the answer the player came for.
    message_focus = kind == 3 ? 0 : 1;
    save_button_rects[0] = save_button_rects[1] = ImRect();
    message_result = -1;
    choice_active = false;
    progress_on = false;
    message_active = true;
    overlay_context->IO.MouseDrawCursor = true;
}

int PollSaveMessage() {
    return message_result;
}

void CloseSaveDialog() {
    std::scoped_lock lock{imgui_mutex};
    if (choice_active.exchange(false)) {
        choice_result = -2;
    }
    if (message_active.exchange(false)) {
        message_result = 2;
    }
    FinishSaveDialog();
}

void SaveProgress(int kind, int active, int percent, const char* text) {
    std::scoped_lock lock{imgui_mutex};
    const auto now = SaveClock::now();
    if (!active) {
        progress_on = false;
        return;
    }
    if (!progress_on && now >= progress_until) {
        progress_since = now;
    }
    progress_kind = kind;
    progress_percent = percent;
    if (text) {
        progress_text = text;
    }
    if (active == 2) {
        progress_until = now + std::chrono::milliseconds(1200);
    } else {
        progress_on = true;
    }
}

void SaveToast(int kind) {
    std::scoped_lock lock{imgui_mutex};
    const auto now = SaveClock::now();
    // A dialog's "saved" and the save's own unmount come together: one toast.
    if (now < toast_until + std::chrono::seconds(3)) {
        toast_until = std::max(toast_until, now + std::chrono::milliseconds(2600));
        return;
    }
    toast_kind = kind;
    toast_since = now;
    toast_until = now + std::chrono::milliseconds(2600);
}

void SetTextEntry(bool active, const std::string& prompt, const std::string& text) {
    std::scoped_lock lock{imgui_mutex};
    text_entry_active = active;
    text_entry_prompt = prompt;
    text_entry_text = text;
}

void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent) {
    const ContextScope context_scope;
    // Present interval for the FPS readout (measured also while nothing is drawn).
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - last_present).count();
    last_present = now;
    if (ms > 0.0f && ms < 1000.0f) {
        frame_ms_avg = frame_ms_avg == 0.0f ? ms : frame_ms_avg * 0.95f + ms * 0.05f;
    }
#ifdef TLG_RENDERER_V2
    { // steam-deck (medição): mesma linha do renderer deste port, a cada 5 s
        static auto window = now;
        static unsigned frames = 0, spikes = 0;
        static float worst = 0.0f;
        if (!StartupImpl::active)
            if (const std::string n = CacaImpl::Frame(ms); !n.empty()) Notice(n, 2.5f);
        if (StartupImpl::active) { // a tela de abertura nao conta como jogo nas medicoes
            window = now;
            frames = spikes = 0;
            worst = 0.0f;
        }
        ++frames;
        if (ms < 1000.0f) {
            worst = std::max(worst, ms);
            spikes += ms > 50.0f;
        }
        const float secs = std::chrono::duration<float>(now - window).count();
        if (secs >= 5.0f) {
            std::printf("Frame stats: %.1f FPS (renderer v2), pior %.1f ms, acima de 50 ms %u\n", frames / secs,
                        worst, spikes);
            std::fflush(stdout);
            frames = 0;
            spikes = 0;
            worst = 0.0f;
            window = now;
        }
    }
#endif
    ConfirmRestart();
    // steam-deck: opened or closed from the pad thread (View button, RequestOpen).
    if (const int request = open_request.exchange(-1, std::memory_order_acq_rel); request >= 0) {
        std::scoped_lock lock{imgui_mutex};
        SetOpen(request != 0);
    }
    if (!Visible()) {
        return;
    }
    std::scoped_lock lock{imgui_mutex};
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(extent.width), float(extent.height));
    io.DeltaTime = ms > 0.0f && ms < 1000.0f ? ms / 1000.0f : 1.0f / 60.0f;
    // UI scale follows the display height (1080p = 1).
    const float scale = std::max(float(extent.height) / 1080.0f, 0.75f);
    if (std::abs(scale - base_scale) > 0.01f) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(scale / base_scale);
        style.FontScaleMain = scale;
        base_scale = scale;
    }

#ifdef TLG_RENDERER_V2
    if (StartupImpl::active) {
        StartupImpl::LoadTextures();
    } else if (StartupImpl::art || StartupImpl::logo) {
        StartupImpl::Retire();
    }
    SkipImpl::UpdateArt();
#endif
    UpdateChoiceTextures();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
#ifdef TLG_RENDERER_V2
    if (StartupImpl::active) {
        StartupImpl::Draw();
    } else if (StartupImpl::tried) {
        // A tela de abertura ja foi mostrada: a antiga nao aparece depois dela.
    } else
#endif
    if (loading) {
        LoadingScreen();
    }
    if (!menu_open) {
        DeckMenuImpl::SteamPause(ImGui::GetForegroundDrawList(), io.DisplaySize.x, io.DisplaySize.y);
    }
    if (menu_open) {
        static const bool classic_env = [] {
            const char* e = std::getenv("TLG_CLASSIC_MENU");
            return e && e[0] == '1';
        }();
        if (classic_env || DeckMenuImpl::classic) {
            ImGui::GetIO().MouseDrawCursor = true;
            Menu();
        } else {
            DeckMenuImpl::Draw();
        }
    } else {
        DeckMenuImpl::shown = false;
        DeckMenuImpl::classic = false;
    }
    if (TlgSettings::Get().show_fps && !menu_open) {
        FpsCounter();
    }
    if (!loading && !menu_open) {
        HealthDisplay();
    }
    if (background_total && !loading && !menu_open) {
        BackgroundLoading();
    }
    if (text_entry_active) {
        TextEntryBox();
    }
    if (!menu_open) {
        SaveDialogs();
    }
    if (NoticeActive() && !menu_open) {
        NoticeBox();
    }
#ifdef TLG_RENDERER_V2
    if (!StartupImpl::active && !menu_open) {
        CacaImpl::DrawCounter();
        SkipImpl::Draw();
    }
#endif
    if (mouse_pad_on && !menu_open) {
        MousePadBox();
    }
    ImGui::Render();

    const vk::RenderingAttachmentInfo attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    {
        // Font atlas uploads submit to the graphics queue themselves.
        std::scoped_lock submit_lock{Vulkan::Scheduler::submit_mutex};
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdbuf);
    }
    cmdbuf.endRendering();
}

} // namespace TlgOverlay
