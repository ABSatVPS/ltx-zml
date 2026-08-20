//! The colibri engine (engine/colibri.c, main renamed coli_engine_main)
//! linked against the Zig/ZML GPU backend — the ZML build of `coli`.
const std = @import("std");

comptime {
    _ = @import("abi.zig"); // emit the coli_cuda_* exports the engine links
}

extern fn coli_engine_main(argc: c_int, argv: [*c][*c]u8) c_int;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const cargv = try arena.alloc([*c]u8, args.len + 1);
    for (args, 0..) |a, i| cargv[i] = @constCast(a.ptr);
    cargv[args.len] = null;
    const rc = coli_engine_main(@intCast(args.len), cargv.ptr);
    if (rc != 0) std.process.exit(@truncate(@as(u32, @bitCast(rc))));
}
