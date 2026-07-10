// ============================================================
// snn_fc_core.v
// Time-multiplexed shared FC + LIF engine with EXTERNAL weight memory.
//
// Difference vs snn_fc_layer.v: weights are NOT held in an internal
// $readmemh reg array (which cannot scale to 135k params on-chip).
// Instead the core drives address ports and consumes data from external
// synchronous memory (SPRAM on iCE40-UP5K) with a 1-cycle read latency.
// This is the realistic architecture: the fc1 matrix (528x256) lives in
// the 4x256Kbit SPRAM banks; the datapath is a single shared LIF unit.
//
// Q6.10 fixed-point, 16-bit signed datapath.
//
// Read protocol (per input index i of output neuron j):
//   S_ADDR : present in_addr=i, w_addr=j*N_IN+i   (data valid next cycle)
//   S_DATA : w_data / in_spike now valid -> if spike, acc += w_data
// Then LIF update per neuron. Bias b[j] read once (b_addr held = j).
// ============================================================
module snn_fc_core #(
  parameter N_IN   = 528,
  parameter N_OUT  = 256,
  parameter AW     = 18,                        // >= clog2(N_IN*N_OUT); 528*256=135168 needs 18
  parameter signed [15:0] BETA   = 16'sd933,    // lif2 0.9108 Q6.10
  parameter signed [15:0] THRESH = 16'sd307     // 0.30   Q6.10
)(
  input  wire clk,
  input  wire rst,
  input  wire start,
  input  wire in_spike,                      // spike bit for in_addr (1-cyc latency)
  output reg  [$clog2(N_IN)-1:0]  in_addr,
  output reg  [AW-1:0]            w_addr,     // j*N_IN + i
  input  wire signed [15:0]       w_data,     // weight[j,i] (1-cyc latency)
  output reg  [$clog2(N_OUT)-1:0] b_addr,     // j
  input  wire signed [15:0]       b_data,     // bias[j]  (Q6.10)
  output reg  [N_OUT-1:0]         spike_out,
  output reg                      done
);
  localparam signed [31:0] SAT_MAX =  32'sd32767;
  localparam signed [31:0] SAT_MIN = -32'sd32768;

  reg signed [15:0] mem_arr [0:N_OUT-1];
  reg               prev_spk [0:N_OUT-1];

  localparam IDLE=3'd0, S_ADDR=3'd1, S_DATA=3'd2, FIRE=3'd3, NEXTN=3'd4, FIN=3'd5;
  reg [2:0] state;
  integer j, i, r;
  reg signed [31:0] acc;

  always @(posedge clk) begin
    if (rst) begin
      for (r=0; r<N_OUT; r=r+1) begin mem_arr[r]<=0; prev_spk[r]<=1'b0; end
      spike_out<=0; done<=0; state<=IDLE; j<=0; i<=0; acc<=0;
      in_addr<=0; w_addr<=0; b_addr<=0;
    end else begin
      case (state)
        IDLE: begin
          done<=0;
          if (start) begin
            j<=0; i<=0; acc<=0;
            in_addr<=0; w_addr<=0; b_addr<=0;
            state<=S_ADDR;
          end
        end
        // address for input i presented; memory returns data next cycle
        S_ADDR: state<=S_DATA;
        S_DATA: begin
          if (in_spike) acc <= acc + w_data;   // AC (spike-driven)
          if (i == N_IN-1) begin
            state<=FIRE;
          end else begin
            i      <= i + 1;
            in_addr<= i + 1;
            w_addr <= j*N_IN + (i + 1);
            state  <= S_ADDR;
          end
        end
        FIRE: begin : lif_calc
          reg signed [31:0] beta_mem, mem_next32;
          reg signed [15:0] leaked;
          beta_mem   = BETA * mem_arr[j];
          leaked     = beta_mem >>> 10;
          mem_next32 = leaked + acc + b_data - (prev_spk[j] ? THRESH : 32'sd0);
          if (mem_next32 > SAT_MAX) mem_next32 = SAT_MAX;
          if (mem_next32 < SAT_MIN) mem_next32 = SAT_MIN;
          mem_arr[j] <= mem_next32[15:0];
          if (mem_next32 >= THRESH) begin spike_out[j]<=1'b1; prev_spk[j]<=1'b1; end
          else                      begin spike_out[j]<=1'b0; prev_spk[j]<=1'b0; end
          state<=NEXTN;
        end
        NEXTN: begin
          if (j == N_OUT-1) state<=FIN;
          else begin
            j      <= j + 1;
            i      <= 0;
            acc    <= 0;
            in_addr<= 0;
            w_addr <= (j + 1)*N_IN;
            b_addr <= (j + 1);
            state  <= S_ADDR;
          end
        end
        FIN: begin done<=1; state<=IDLE; end
      endcase
    end
  end
endmodule
