// ============================================================
// spram_conv1_weights.v
// Proves the SB_SPRAM256KA integration pattern: boot-load real
// trained weights into the hard-macro SPRAM at reset, then serve
// them as a synchronous ROM (1-cycle read latency, matching the
// convention every core module already expects on w_addr/w_data).
//
// IMPORTANT SCOPE NOTE: conv1's table (336 words) is far smaller
// than one 256Kbit SPRAM bank (16384 words) - this is deliberately
// NOT the final placement for conv1 (which belongs in a small EBR
// or LUT-ROM). This module exists to PROVE the primitive's usage
// pattern (boot-load write sequence, address decode, read timing)
// on real, convenient weight data, before scaling the same proven
// pattern to fc1's actual 4-bank + EBR-overflow mapping.
//
// Unlike generic behavioral memories, SB_SPRAM256KA has NO
// bitstream-initialization parameters (confirmed from cells_sim.v -
// no INIT_* ports). Real hardware must therefore perform an active
// write sequence after every power-up/reset before first use - this
// module implements exactly that boot-load sequence.
// ============================================================
module spram_conv1_weights #(
  parameter N_W = 336   // conv1: 320 weights + 16 biases, packed together
)(
  input  wire clk,
  input  wire rst,
  input  wire [8:0]  rd_addr,     // 0..335
  output wire signed [15:0] rd_data,
  output reg  boot_done
);
  // ---- boot-load ROM: the actual trained values, known at synth time ----
  reg signed [15:0] boot_rom [0:N_W-1];
  initial $readmemh("conv1_all_q610.hex", boot_rom);  // weights[0:319] then biases[320:335]

  reg [8:0] boot_addr;
  reg       boot_wren;

  wire [13:0] spram_addr = boot_wren ? {5'b0, boot_addr} : {5'b0, rd_addr};
  wire [15:0] spram_din  = boot_rom[boot_addr];
  wire [15:0] spram_dout;

  SB_SPRAM256KA u_spram (
    .DATAIN(spram_din),
    .ADDRESS(spram_addr),
    .MASKWREN(4'b1111),
    .WREN(boot_wren),
    .CHIPSELECT(1'b1),
    .CLOCK(clk),
    .STANDBY(1'b0),
    .SLEEP(1'b0),
    .POWEROFF(1'b1),        // 1 = normal operation (NOT powered off)
    .DATAOUT(spram_dout)
  );

  assign rd_data = spram_dout;

  always @(posedge clk) begin
    if (rst) begin
      boot_addr <= 0; boot_wren <= 1'b1; boot_done <= 1'b0;
    end else if (boot_wren) begin
      if (boot_addr == N_W-1) begin
        boot_wren <= 1'b0; boot_done <= 1'b1;
      end else begin
        boot_addr <= boot_addr + 1;
      end
    end
  end
endmodule
