`timescale 1ns/1ps
module lat_tb;
  reg clk=0, rst=1, start=0;
  reg [131:0] conv1_in;
  wire [1:0] fc2_spike;
  wire signed [15:0] m0,m1;
  wire done;
  integer cyc=0, t0=0;
  snn_top dut(.clk(clk),.rst(rst),.start(start),.conv1_in(conv1_in),
              .fc2_spike(fc2_spike),.fc2_mem0(m0),.fc2_mem1(m1),.done(done));
  always #5 clk=~clk;
  always @(posedge clk) cyc<=cyc+1;
  initial begin
    conv1_in = 132'h1; // arbitrary stimulus, we only measure latency
    repeat(6) @(posedge clk);
    rst=0;
    // wait for the reset CLR passes to settle
    repeat(600) @(posedge clk);
    @(posedge clk) start=1; t0=cyc;
    @(posedge clk) start=0;
    wait(done);
    $display("ONE_TIMESTEP_CYCLES=%0d", cyc - t0);
    $display("FULL_INFERENCE_197_CYCLES=%0d", (cyc - t0)*197);
    $finish;
  end
endmodule
