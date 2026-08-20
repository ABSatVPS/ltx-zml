//! Runs upstream's conformance test (tests/test_backend_cuda.cc, one-line
//! patch: its main is renamed coli_test_main) against the Zig coli_cuda ABI.
const std = @import("std");

comptime {
    _ = @import("abi.zig"); // force emission of the coli_cuda_* exports
}

extern fn coli_test_main(argc: c_int, argv: [*c][*c]u8) c_int;

pub fn main(init: std.process.Init) !void {
    _ = init;
    var name: [12:0]u8 = "conformance\x00".*;
    var argv = [_][*c]u8{ &name, null };
    const rc = coli_test_main(1, &argv);
    if (rc == 77) {
        std.log.err("SKIP: no usable device (coli_cuda_init failed)", .{});
        std.process.exit(77);
    }
    if (rc != 0) {
        std.log.err("conformance FAILED with code {d}", .{rc});
        std.process.exit(@intCast(@as(u8, @truncate(@as(u32, @bitCast(rc))))));
    }
    std.log.info("conformance PASS", .{});
}
