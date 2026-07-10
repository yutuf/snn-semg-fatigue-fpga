`timescale 1ns/1ps
module snn_fc_tb;
  localparam N_IN=4, N_OUT=2;
  reg clk=0, rst=1, start=0;
  reg [N_IN-1:0] spike_in=0;
  wire [N_OUT-1:0] spike_out;
  wire done;

  snn_fc_layer #(.N_IN(N_IN), .N_OUT(N_OUT),
                 .BETA(16'sd870), .THRESH(16'sd307), .WFILE("weights.hex"))
    uut(.clk(clk),.rst(rst),.start(start),.spike_in(spike_in),.spike_out(spike_out),.done(done));

  always #5 clk=~clk;

  task run_step; begin
    @(negedge clk); start=1;
    @(negedge clk); start=0;
    wait(done==1);
    @(negedge clk);
    $display("  spike_out=%b | mem[0]=%0d (%.3f) mem[1]=%0d (%.3f)",
      spike_out, uut.mem_arr[0], uut.mem_arr[0]/1024.0, uut.mem_arr[1], uut.mem_arr[1]/1024.0);
  end endtask

  initial begin
    repeat(3) @(negedge clk); rst=0;
    spike_in=4'b1111;   // tum girisler aktif
    $display("--- Zaman adimi 1 (beklenen: spike=11, mem0=2048/2.0, mem1=408/0.398) ---");
    run_step;
    $display("--- Zaman adimi 2 (beklenen: spike=11, mem0=3481/3.4, mem1=447/0.436) ---");
    run_step;
    $finish;
  end
endmodule
