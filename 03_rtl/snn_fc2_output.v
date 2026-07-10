// ============================================================
// snn_fc2_output.v
// Output layer (fc2, 256->2). Unlike the hidden layers, classification
// uses the MEMBRANE POTENTIAL averaged over time (out = mem.mean(0) in
// training), not the spike count - so this variant exposes the final
// membrane value per output neuron, not just spike_out.
// N_OUT=2 is tiny: membranes kept in plain registers (no BRAM needed -
// the BRAM lesson only matters once N_OUT gets into the hundreds, as
// with fc1's 256 or conv1's 528).
// Weights/bias still stream from external memory (same convention as
// the other layers), since fc2 still has 256*2+2=514 parameters.
// Q6.10 fixed-point, 16-bit signed datapath.
// ============================================================
module snn_fc2_output #(
  parameter N_IN   = 256,
  parameter N_OUT  = 2,
  parameter signed [15:0] BETA   = 16'sd907,  // lif3 0.88531 Q6.10
  parameter signed [15:0] THRESH = 16'sd307   // 0.30 Q6.10
)(
  input  wire clk,
  input  wire rst,
  input  wire start,
  input  wire in_spike,
  output reg  [$clog2(N_IN)-1:0]  in_addr,
  output reg  [$clog2(N_IN*N_OUT)-1:0] w_addr,
  input  wire signed [15:0]       w_data,
  output reg  [$clog2(N_OUT)-1:0] b_addr,
  input  wire signed [15:0]       b_data,
  output reg  [N_OUT-1:0]         spike_out,
  output reg  signed [16*N_OUT-1:0] mem_out_flat,  // packed: neuron j = mem_out_flat[16*j +: 16]
  output reg                      done
);
  localparam signed [31:0] SAT_MAX =  32'sd32767;
  localparam signed [31:0] SAT_MIN = -32'sd32768;

  reg signed [15:0] mem_arr [0:N_OUT-1];
  reg               prev_spk [0:N_OUT-1];

  localparam IDLE=2'd0, ACC=2'd1, FIRE=2'd2, FIN=2'd3;
  reg [1:0] state;
  integer j, i, r;
  reg signed [31:0] acc;

  always @(posedge clk) begin
    if (rst) begin
      for (r=0; r<N_OUT; r=r+1) begin
        mem_arr[r] <= 0; prev_spk[r] <= 1'b0;
      end
      mem_out_flat <= 0;
      spike_out <= 0; done <= 0; state <= IDLE; j <= 0; i <= 0; acc <= 0;
      in_addr<=0; w_addr<=0; b_addr<=0;
    end else begin
      case (state)
        IDLE: begin
          done <= 0;
          if (start) begin
            j<=0; i<=0; acc<=0; in_addr<=0; w_addr<=0; b_addr<=0; state<=ACC;
          end
        end
        ACC: begin
          if (in_spike) acc <= acc + w_data;
          if (i == N_IN-1) state <= FIRE;
          else begin i <= i+1; in_addr <= i+1; w_addr <= j*N_IN + (i+1); end
        end
        FIRE: begin : lif_calc
          reg signed [31:0] beta_mem, mem_next32;
          beta_mem   = BETA * mem_arr[j];
          mem_next32 = (beta_mem >>> 10) + acc + b_data - (prev_spk[j] ? THRESH : 32'sd0);
          if (mem_next32 > SAT_MAX) mem_next32 = SAT_MAX;
          if (mem_next32 < SAT_MIN) mem_next32 = SAT_MIN;
          mem_arr[j] <= mem_next32[15:0];
          mem_out_flat[16*j +: 16] <= mem_next32[15:0];  // expose it - classification reads this
          if (mem_next32 >= THRESH) begin spike_out[j]<=1'b1; prev_spk[j]<=1'b1; end
          else                      begin spike_out[j]<=1'b0; prev_spk[j]<=1'b0; end
          if (j == N_OUT-1) state <= FIN;
          else begin j <= j+1; i <= 0; acc <= 0; b_addr <= j+1;
                     in_addr <= 0; w_addr <= (j+1)*N_IN; state <= ACC; end
        end
        FIN: begin done <= 1; state <= IDLE; end
      endcase
    end
  end
endmodule
