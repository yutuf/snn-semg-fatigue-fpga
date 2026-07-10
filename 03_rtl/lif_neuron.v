module lif_neuron #(
  parameter signed [15:0] BETA   = 16'sd841,  // 0.8216 (lif1) in Q6.10
  parameter signed [15:0] THRESH = 16'sd307   // 0.30 in Q6.10
)(
  input  wire clk, input wire rst, input wire en,
  input  wire signed [15:0] i_current,   // Q6.10 agirlikli giris toplami
  output reg  o_spike,
  output reg  signed [15:0] o_mem
);
  reg signed [15:0] mem;
  reg prev_spike;
  wire signed [31:0] beta_mem = BETA * mem;                 // Q12.20
  wire signed [15:0] leaked   = beta_mem >>> 10;            // Q6.10'a geri
  wire signed [15:0] mem_next = leaked + i_current - (prev_spike ? THRESH : 16'sd0);
  wire spike_next = (mem_next >= THRESH);
  always @(posedge clk) begin
    if (rst)      begin mem<=0; prev_spike<=0; o_spike<=0; o_mem<=0; end
    else if (en)  begin mem<=mem_next; prev_spike<=spike_next; o_spike<=spike_next; o_mem<=mem_next; end
  end
endmodule
