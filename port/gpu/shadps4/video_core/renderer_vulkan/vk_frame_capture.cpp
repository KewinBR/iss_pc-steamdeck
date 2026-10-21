// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_frame_capture.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <format>
#include <mutex>

#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/texture_cache/image_info.h"
#include "core/memory.h"
#include <unordered_set>

extern "C" void tlggpu_overlay_notice(const char* text, float seconds);

namespace Vulkan {

namespace {

struct Target {
    VAddr address{};
    vk::Format format{};
    u32 width{}, height{};
    bool operator==(const Target&) const = default;
};

struct Entry {
    bool compute{};
    std::vector<Target> colors;
    Target depth;
    u32 draws{};
    u64 indices{};
    std::vector<u64> shaders; ///< vs/ps pairs (graphics) or cs (compute), first few
    std::vector<std::pair<Target, bool>> sampled; ///< textures (false) and storage images (true)
    std::string state; ///< census line of the first pipeline (vk_pass_census.h)
    std::string note;
    std::vector<std::string> buffers; ///< dumped constants, per draw
    /// steam-deck: desenhos por par vs/ps no passe (até 64 pares; o resto conta em pair_other).
    std::vector<std::pair<std::pair<u64, u64>, u32>> pairs;
    u32 pair_other{};
};

// GPU thread only.
std::vector<Entry> entries;
std::vector<std::pair<Target, bool>> pending_sampled;
std::vector<std::string> pending_buffers;

// steam-deck: modo inspeção (Inspect, no .h). inspect_draws: GPU thread enquanto grava; depois que a gravação
// termina, só o VideoOut lê (no InspectResolve, 3 flips depois). Outra inspeção só arma depois disso.
constexpr u32 kInspectMax = 16384;
struct InspectDraw {
    Inspect::DrawInfo info;
    u32 pass{}, draw{};
    std::string textures; ///< as texturas do desenho (pending_sampled no Begin: ligadas antes dele)
};
std::vector<InspectDraw> inspect_draws;
std::atomic<bool> inspect_pending{false};
std::atomic<u64> inspect_resolve_at{0};
vk::Device inspect_device{};
std::string inspect_capture; ///< arquivo da captura do mesmo quadro

void AddBuffers(Entry& entry) {
    // Constants of full-screen passes (post-processing: camera matrices) and of the first draw
    // of each geometry pass.
    if (entry.draws <= 4) {
        for (auto& b : pending_buffers) {
            entry.buffers.push_back(std::move(b));
        }
    }
    pending_buffers.clear();
}
bool pass_open = false;
std::mutex display_mutex;
std::vector<VAddr> display_buffers;
// tlg: read on every draw; the game registers a handful of display buffers once.
std::array<std::atomic<VAddr>, 16> display_list{};
std::atomic<u32> display_count{0};

Target ToTarget(const VideoCore::ImageInfo& info) {
    return {info.guest_address, info.pixel_format, info.size.width, info.size.height};
}

std::string Describe(const Target& t) {
    if (!t.address) {
        return "-";
    }
    return std::format("{:#x} {} {}x{}", t.address, vk::to_string(t.format), t.width, t.height);
}

void AddSampled(Entry& entry) {
    for (const auto& s : pending_sampled) {
        if (entry.sampled.size() < 32 &&
            std::ranges::find(entry.sampled, s) == entry.sampled.end()) {
            entry.sampled.push_back(s);
        }
    }
    pending_sampled.clear();
}

void AddShader(Entry& entry, u64 hash) {
    if (entry.shaders.size() < 8 && std::ranges::find(entry.shaders, hash) == entry.shaders.end()) {
        entry.shaders.push_back(hash);
    }
}

std::mutex result_mutex;
std::string result; ///< for the menu (FrameCapture::LastResult)

void SetResult(std::string text) {
    std::printf("Frame capture: %s\n", text.c_str());
    std::scoped_lock lk{result_mutex};
    result = std::move(text);
}

// TLG_CAPTURE_DIR, else <user>/captures (TLG_GPU_USER_DIR, the saves folder).
std::filesystem::path CaptureDir() {
    if (const char* dir = std::getenv("TLG_CAPTURE_DIR"); dir && dir[0]) {
        return dir;
    }
    const char* user = std::getenv("TLG_GPU_USER_DIR");
    return std::filesystem::path(user && user[0] ? user : "user") / "captures";
}

void Write(VAddr presented) {
    const auto dir = CaptureDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    static u32 sequence = 0;
    const std::string path = (dir / std::format("frame_{}_{}.txt", stamp, ++sequence)).string();
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) {
        std::printf("Frame capture: cannot write %s\n", path.c_str());
        SetResult("Capture failed: cannot write " + path);
        return;
    }
    std::fprintf(f, "presented buffer %#llx\n", static_cast<unsigned long long>(presented));
    u32 index = 0;
    for (const auto& e : entries) {
        if (e.compute) {
            std::fprintf(f, "\n#%u COMPUTE dispatches %u, cs", index++, e.draws);
        } else {
            std::fprintf(f, "\n#%u PASS draws %u (indices %llu), vs/ps", index++, e.draws,
                         static_cast<unsigned long long>(e.indices));
        }
        for (const u64 h : e.shaders) {
            std::fprintf(f, " %016llx", static_cast<unsigned long long>(h));
        }
        std::fprintf(f, "\n");
        for (size_t i = 0; i < e.colors.size(); ++i) {
            std::fprintf(f, "  color%zu %s\n", i, Describe(e.colors[i]).c_str());
        }
        if (!e.compute) {
            std::fprintf(f, "  depth  %s\n", Describe(e.depth).c_str());
        }
        if (!e.state.empty()) {
            std::fprintf(f, "  %s\n", e.state.c_str());
        }
        if (e.pairs.size() > 1 || e.pair_other) {
            auto pairs = e.pairs;
            std::ranges::sort(pairs, std::greater{}, &std::pair<std::pair<u64, u64>, u32>::second);
            std::fprintf(f, "  pares vs/ps (desenhos):");
            for (size_t i = 0; i < pairs.size() && i < 16; ++i) {
                std::fprintf(f, " %08x/%08x:%u", u32(pairs[i].first.first),
                             u32(pairs[i].first.second), pairs[i].second);
            }
            if (pairs.size() > 16 || e.pair_other) {
                u32 rest = e.pair_other;
                for (size_t i = 16; i < pairs.size(); ++i) {
                    rest += pairs[i].second;
                }
                std::fprintf(f, " outros:%u", rest);
            }
            std::fprintf(f, "\n");
        }
        for (const auto& [t, storage] : e.sampled) {
            std::fprintf(f, "  %s %s\n", storage ? "writes " : "samples", Describe(t).c_str());
        }
        if (!e.note.empty()) {
            std::fprintf(f, "  note: %s\n", e.note.c_str());
        }
        for (const auto& b : e.buffers) {
            std::fprintf(f, "%s\n", b.c_str());
        }
    }
    std::fclose(f);
    inspect_capture = path;
    std::printf("Frame capture: %zu passes written to %s\n", entries.size(), path.c_str());
    SetResult(std::format("Frame captured ({} passes): {}", entries.size(), path));
}

u32 InspectCode(const Inspect::DrawInfo& d) {
    u64 h = 1469598103934665603ull;
    for (const u64 v : {u64(d.indices), u64(d.num_indices), d.vs, d.ps}) {
        h = (h ^ v) * 1099511628211ull;
    }
    return u32(h ^ (h >> 32)) & 0xffff;
}

void InspectResolve() {
    const u32 count = u32(inspect_draws.size());
    std::vector<u64> samples(count);
    if (count) {
        const auto r = inspect_device.getQueryPoolResults(
            Inspect::Pool(), 0, count, count * sizeof(u64), samples.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (r != vk::Result::eSuccess) {
            std::printf("Inspect: query results failed\n");
            inspect_draws.clear();
            return;
        }
    }
    // Geometria de verdade: com índices (os triângulos de tela cheia não têm), no alvo de maior
    // área entre os candidatos (a cena; o minimapa desenha num 768x768 depois dela).
    const auto candidate = [&](u32 i) {
        const auto& d = inspect_draws[i].info;
        return samples[i] && d.color && d.depth_write && d.indices;
    };
    u32 best_area = 0;
    for (u32 i = 0; i < count; ++i) {
        if (candidate(i)) {
            best_area = std::max(best_area, u32(inspect_draws[i].info.width) * inspect_draws[i].info.height);
        }
    }
    int pick = -1;
    for (u32 i = 0; i < count; ++i) {
        if (candidate(i) && u32(inspect_draws[i].info.width) * inspect_draws[i].info.height == best_area) {
            pick = int(i);
        }
    }
    const auto dir = CaptureDir();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&now));
    const std::string path = (dir / std::format("inspecao_{}.txt", stamp)).string();
    std::string headline = "Inspecao: nada com profundidade no centro da tela";
    if (pick >= 0) {
        const auto& p = inspect_draws[pick];
        headline = std::format("Inspecao: M-{:04X} (passe {} desenho {}, {} indices)", InspectCode(p.info),
                               p.pass, p.draw, p.info.num_indices);
    }
    if (FILE* f = std::fopen(path.c_str(), "w")) {
        std::fprintf(f, "%s\ncaptura do mesmo quadro: %s\n%u desenhos inspecionados; os que passaram pelo centro "
                        "(> = o objeto mirado: o ultimo com cor e profundidade):\n\n",
                     headline.c_str(), inspect_capture.c_str(), count);
        for (u32 i = 0; i < count; ++i) {
            if (!samples[i]) {
                continue;
            }
            const auto& d = inspect_draws[i];
            std::fprintf(f, "%c M-%04X #%u.%u vs %016llx ps %016llx indices %#llx n %u inst %u v0 %u i0 %u z %u cor %u "
                            "color0 %#llx %ux%u amostras %llu\n",
                         int(i) == pick ? '>' : ' ', InspectCode(d.info), d.pass, d.draw,
                         (unsigned long long)d.info.vs, (unsigned long long)d.info.ps,
                         (unsigned long long)d.info.indices, d.info.num_indices, d.info.instances,
                         d.info.first_vertex, d.info.first_instance, u32(d.info.depth_write), u32(d.info.color),
                         (unsigned long long)d.info.color0, d.info.width, d.info.height,
                         (unsigned long long)samples[i]);
            if (int(i) == pick || d.info.indices) {
                std::fprintf(f, "%s", d.textures.c_str());
            }
        }
        std::fclose(f);
    }
    std::printf("%s -> %s\n", headline.c_str(), path.c_str());
    tlggpu_overlay_notice(headline.c_str(), 12.0f);
    inspect_draws.clear();
}

} // namespace

u32 Inspect::Begin(vk::Device device, const DrawInfo& info) {
    if (entries.empty() || inspect_draws.size() >= kInspectMax) {
        return ~0u;
    }
    if (!pool) {
        inspect_device = device;
        const vk::QueryPoolCreateInfo create = {
            .queryType = vk::QueryType::eOcclusion,
            .queryCount = kInspectMax,
        };
        pool = Check<"create occlusion query pool">(device.createQueryPool(create));
    }
    if (inspect_draws.empty()) {
        device.resetQueryPool(pool, 0, kInspectMax);
    }
    std::string textures;
    for (const auto& [t, storage] : pending_sampled) {
        textures += std::format("      {} {}\n", storage ? "escreve" : "textura", Describe(t));
    }
    inspect_draws.push_back({info, u32(entries.size() - 1), entries.back().draws + 1, std::move(textures)});
    return u32(inspect_draws.size() - 1);
}

void FrameCapture::OnFlip(VAddr presented_address) {
    last_presented.store(presented_address, std::memory_order_relaxed);
    const u64 flip = flips.fetch_add(1, std::memory_order_release) + 1;
    static const char* trigger = std::getenv("TLG_CAPTURE_TRIGGER");
    static const char* inspect_trigger = std::getenv("TLG_INSPECT_TRIGGER");
    if (const u64 at = inspect_resolve_at.load(std::memory_order_acquire); at && flip >= at) {
        inspect_resolve_at.store(0, std::memory_order_relaxed);
        InspectResolve();
    }
    if (state.load(std::memory_order_relaxed) != Idle) {
        return;
    }
    bool arm = requested.exchange(false, std::memory_order_acq_rel);
    if (!arm && trigger && std::filesystem::exists(trigger)) {
        std::error_code ec;
        std::filesystem::remove(trigger, ec);
        arm = true;
    }
    if (!arm && inspect_trigger && inspect_trigger[0] &&
        inspect_resolve_at.load(std::memory_order_acquire) == 0 &&
        std::filesystem::exists(inspect_trigger)) {
        std::error_code ec;
        std::filesystem::remove(inspect_trigger, ec);
        arm = true;
        inspect_pending.store(true, std::memory_order_release);
        tlggpu_overlay_notice("Inspecao: lendo o centro da tela...", 2.0f);
    }
    if (arm) {
        state.store(Armed, std::memory_order_release);
        std::printf("Frame capture: armed\n");
    }
}

void FrameCapture::Request() {
    SetResult("Capturing the next frame...");
    requested.store(true, std::memory_order_release);
}

std::string FrameCapture::LastResult() {
    std::scoped_lock lk{result_mutex};
    return result;
}

void FrameCapture::Poll() {}

bool FrameCapture::IsDisplayBuffer(VAddr address) {
    const u32 count = display_count.load(std::memory_order_acquire);
    for (u32 i = 0; i < count; ++i) {
        if (display_list[i].load(std::memory_order_relaxed) == address) {
            return true;
        }
    }
    if (count < display_list.size()) {
        return false;
    }
    std::scoped_lock lk{display_mutex};
    return std::ranges::find(display_buffers, address) != display_buffers.end();
}

void FrameCapture::AddDisplayBuffer(VAddr address) {
    std::scoped_lock lk{display_mutex};
    if (std::ranges::find(display_buffers, address) == display_buffers.end()) {
        display_buffers.push_back(address);
        const u32 count = display_count.load(std::memory_order_relaxed);
        if (count < display_list.size()) {
            display_list[count].store(address, std::memory_order_relaxed);
            display_count.store(count + 1, std::memory_order_release);
        }
    }
}

void FrameCapture::BeginPass(const VideoCore::ImageInfo* const* colors, u32 num_colors,
                             const VideoCore::ImageInfo* depth, std::string census) {
    std::vector<Target> targets;
    bool display = false;
    {
        std::scoped_lock lk{display_mutex};
        for (u32 i = 0; i < num_colors; ++i) {
            targets.push_back(colors[i] ? ToTarget(*colors[i]) : Target{});
            display |= colors[i] && std::ranges::find(display_buffers, colors[i]->guest_address) !=
                                        display_buffers.end();
        }
    }
    // Frame boundaries in the command stream: the pass that writes a display buffer as its only
    // target. tlg: God of War III also draws scene passes (two targets) into display buffers.
    display = display && targets.size() == 1;
    const bool new_pass = !(pass_open && !entries.empty() && !entries.back().compute &&
                            entries.back().colors == targets);
    if (display && new_pass) {
        if (state.load(std::memory_order_relaxed) == Recording) {
            Write(last_presented.load(std::memory_order_relaxed));
            entries.clear();
            pending_sampled.clear();
            pending_buffers.clear();
            if (Inspect::active.exchange(false, std::memory_order_relaxed)) {
                inspect_resolve_at.store(flips.load(std::memory_order_relaxed) + 3, std::memory_order_release);
            }
            state.store(Idle, std::memory_order_release);
            return;
        }
        entries.clear();
        pending_sampled.clear();
        pending_buffers.clear();
        pass_open = false;
        if (inspect_pending.exchange(false, std::memory_order_acq_rel)) {
            inspect_draws.clear();
            Inspect::active.store(true, std::memory_order_relaxed);
        }
        state.store(Recording, std::memory_order_release);
    }
    if (state.load(std::memory_order_relaxed) != Recording) {
        pending_sampled.clear();
        pending_buffers.clear();
        return;
    }
    const Target d = depth ? ToTarget(*depth) : Target{};
    if (pass_open && !entries.empty() && !entries.back().compute &&
        entries.back().colors == targets && entries.back().depth == d) {
        return;
    }
    entries.push_back(
        {.compute = false, .colors = std::move(targets), .depth = d, .state = std::move(census)});
    pass_open = true;
}

void FrameCapture::Draw(u64 vs_hash, u64 ps_hash, u32 num_indices, u32 num_instances) {
    if (state.load(std::memory_order_relaxed) != Recording || entries.empty() ||
        entries.back().compute) {
        return;
    }
    auto& e = entries.back();
    ++e.draws;
    e.indices += u64(num_indices) * std::max(num_instances, 1u);
    AddShader(e, vs_hash);
    AddShader(e, ps_hash);
    if (auto it = std::ranges::find(e.pairs, std::pair{vs_hash, ps_hash},
                                    &std::pair<std::pair<u64, u64>, u32>::first);
        it != e.pairs.end()) {
        ++it->second;
    } else if (e.pairs.size() < 64) {
        e.pairs.push_back({{vs_hash, ps_hash}, 1});
    } else {
        ++e.pair_other;
    }
    AddSampled(e);
    AddBuffers(e);
}

void FrameCapture::Dispatch(u64 cs_hash, u32 x, u32 y, u32 z) {
    if (state.load(std::memory_order_relaxed) != Recording) {
        pending_sampled.clear();
        pending_buffers.clear();
        return;
    }
    if (!entries.empty() && entries.back().compute && !entries.back().shaders.empty() &&
        entries.back().shaders[0] == cs_hash) {
        ++entries.back().draws;
    } else {
        entries.push_back({.compute = true, .draws = 1, .shaders = {cs_hash}});
        entries.back().note = std::format("groups {}x{}x{}", x, y, z);
    }
    AddSampled(entries.back());
    AddBuffers(entries.back());
    pass_open = false;
}

void FrameCapture::Buffer(u64 stage_hash, u32 slot, VAddr address, const void* data, u64 size) {
    if (!data) {
        return;
    }
    const u64 bytes = std::min<u64>(size, 1024) & ~u64(3);
    std::string out = std::format("  buffer stage {:016x} slot {} at {:#x} size {}:", stage_hash,
                                  slot, address, size);
    const auto* words = static_cast<const float*>(data);
    for (u64 i = 0; i < bytes / 4; ++i) {
        if (i % 8 == 0) {
            out += std::format("\n    [{:3}]", i);
        }
        out += std::format(" {:12.6g}", words[i]);
    }
    pending_buffers.push_back(std::move(out));
}

void FrameCapture::Sampled(const VideoCore::ImageInfo& info, bool storage) {
    if (pending_sampled.size() < 64) {
        pending_sampled.emplace_back(ToTarget(info), storage);
    }
    // steam-deck: os bytes crus (memória do jogo) de cada textura lida no quadro gravado, até 2 MiB cada e 96 MiB
    // por captura (TLG_CAPTURE_TEX_MB), em <captures>/tex/<endereço>_<LxA>_<formato>.bin: pra achar e trocar texturas nos pacotes.
    static std::unordered_set<VAddr> dumped;
    static u64 dumped_bytes = 0;
    static u32 dumped_capture = ~0u;
    if (state.load(std::memory_order_relaxed) != Recording || storage || !info.guest_address ||
        info.guest_size == 0 || info.guest_size > (2u << 20)) {
        return;
    }
    const u32 capture = u32(flips.load(std::memory_order_relaxed) >> 8);
    if (capture != dumped_capture) {
        dumped_capture = capture;
        dumped.clear();
        dumped_bytes = 0;
    }
    // TLG_CAPTURE_TEX_MB: teto por captura (96 MiB padrão; os menus por cima do mundo pedem mais).
    static const u64 limit = [] {
        const char* e = std::getenv("TLG_CAPTURE_TEX_MB");
        return (e && *e ? std::strtoull(e, nullptr, 10) : 96ull) << 20;
    }();
    if (dumped_bytes > limit || !dumped.insert(info.guest_address).second) {
        return;
    }
    auto* memory = Core::Memory::Instance();
    if (!memory->IsValidGpuMapping(info.guest_address, info.guest_size) ||
        memory->ClampRangeSize(info.guest_address, info.guest_size) != info.guest_size) {
        return;
    }
    const auto dir = CaptureDir() / "tex";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / std::format("{:x}_{}x{}_{}.bin", info.guest_address, info.size.width,
                                        info.size.height, vk::to_string(info.pixel_format));
    if (FILE* f = std::fopen(path.string().c_str(), "wb")) {
        std::fwrite(reinterpret_cast<const void*>(info.guest_address), 1, info.guest_size, f);
        std::fclose(f);
        dumped_bytes += info.guest_size;
    }
}

void FrameCapture::Note(const char* text) {
    if (!entries.empty()) {
        entries.back().note += text;
    }
}

} // namespace Vulkan
