`timescale 1ns/1ps
module lif_tb;
  reg clk=0, rst=1, en=0; reg signed [15:0] i_current=0;
  wire o_spike; wire signed [15:0] o_mem;
  lif_neuron uut(.clk(clk),.rst(rst),.en(en),.i_current(i_current),.o_spike(o_spike),.o_mem(o_mem));
  always #5 clk=~clk;
  integer k;
  initial begin
    #12 rst=0; en=1; i_current=16'sd102;   // 0.1 sabit giris (Q6.10)
    for (k=0;k<40;k=k+1) begin
      @(posedge clk);
      $display("t=%2d  mem=%5d (%f)  spike=%b", k, o_mem, o_mem/1024.0, o_spike);
    end
    $finish;
  end
endmodule
