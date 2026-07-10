`timescale 1ns/1ps
// Proves the SB_SPRAM256KA boot-load-then-serve pattern: after reset,
// the module writes all 336 real conv1 weights/biases into the hard-macro
// SPRAM, then every address must read back bit-exact against the source.
module spram_conv1_weights_tb;
  reg clk=0, rst=1;
  reg [8:0] rd_addr=0;
  wire signed [15:0] rd_data;
  wire boot_done;

  spram_conv1_weights uut(.clk(clk), .rst(rst), .rd_addr(rd_addr),
                           .rd_data(rd_data), .boot_done(boot_done));

  always #5 clk=~clk;

  reg signed [15:0] expected [0:335];
  integer i, errors;

  initial begin
    $readmemh("conv1_all_q610.hex", expected);

    repeat(3) @(negedge clk); rst=0;
    wait(boot_done==1);
    repeat(3) @(negedge clk);
    $display("Boot-load done after reset. Reading back all 336 entries...");

    errors=0;
    for (i=0;i<336;i=i+1) begin
      rd_addr = i[8:0];
      @(negedge clk); @(negedge clk);   // SPRAM read is registered (1-cycle latency)
      if (rd_data !== expected[i]) begin
        errors = errors+1;
        if (errors<=5) $display("  mismatch addr=%0d got=%0d expected=%0d", i, rd_data, expected[i]);
      end
    end

    if (errors==0) $display("RESULT: PASS - SPRAM boot-load + readback bit-exact on all 336 entries");
    else           $display("RESULT: FAIL - %0d mismatches", errors);
    $finish;
  end
endmodule
