// ============================================================
// snn_fc_core_bram.v
// Time-multiplexed shared FC + LIF engine, BRAM-backed membranes.
//
// Same datapath as snn_fc_core.v, but the N_OUT membrane potentials are
// held in an inferred block-RAM (EBR) instead of a fabric flip-flop file.
// This removes ~N_OUT*16 flip-flops and their read/write mux LUTs, which
// (per the full-scale synthesis finding) otherwise nearly fill the UP5K.
//
// Weights + biases stream from external synchronous memory (SPRAM),
// 1-cycle read latency. Membrane BRAM also has 1-cycle synchronous read.
// Q6.10 fixed-point, 16-bit signed datapath.
// ============================================================
module snn_fc_core_bram #(
  parameter N_IN   = 528,
  parameter N_OUT  = 256,
  parameter AW     = 18,                        // >= clog2(N_IN*N_OUT)
  parameter signed [15:0] BETA   = 16'sd933,    // lif2 0.9108 Q6.10
  parameter signed [15:0] THRESH = 16'sd307     // 0.30   Q6.10
)(
  input  wire clk,
  input  wire rst,
  input  wire start,
  input  wire in_spike,
  output reg  [$clog2(N_IN)-1:0]  in_addr,
  output reg  [AW-1:0]            w_addr,
  input  wire signed [15:0]       w_data,
  output reg  [$clog2(N_OUT)-1:0] b_addr,
  input  wire signed [15:0]       b_data,
  output reg  [N_OUT-1:0]         spike_out,
  output reg                      done
);
  localparam signed [31:0] SAT_MAX =  32'sd32767;
  localparam signed [31:0] SAT_MIN = -32'sd32768;

  // ---- membrane BRAM (inferred EBR): synchronous read + write ----
  reg signed [15:0] mem_ram [0:N_OUT-1];
  reg [$clog2(N_OUT)-1:0] mem_ra, mem_wa;
  reg               mem_we;
  reg signed [15:0] mem_wd, mem_rd;
  always @(posedge clk) begin
    if (mem_we) mem_ram[mem_wa] <= mem_wd;
    mem_rd <= mem_ram[mem_ra];
  end

  // prev-spike stays in FFs (only N_OUT bits, cheap)
  reg prev_spk [0:N_OUT-1];

  localparam CLR=3'd0, IDLE=3'd1, MEMRD=3'd2, S_ADDR=3'd3,
             S_DATA=3'd4, FIRE=3'd5, NEXTN=3'd6, FIN=3'd7;
  reg [2:0] state;
  integer j, i, r;
  reg signed [31:0] acc;
  reg [$clog2(N_OUT):0] clr_i;

  always @(posedge clk) begin
    if (rst) begin
      // begin a clear pass over the membrane BRAM + prev_spk
      spike_out<=0; done<=0; j<=0; i<=0; acc<=0;
      in_addr<=0; w_addr<=0; b_addr<=0;
      clr_i<=0; mem_we<=1'b1; mem_wa<=0; mem_wd<=16'sd0;
      state<=CLR;
    end else begin
      case (state)
        CLR: begin
          // walk all membranes writing 0; clear prev_spk
          mem_we<=1'b1; mem_wa<=clr_i[$clog2(N_OUT)-1:0]; mem_wd<=16'sd0;
          prev_spk[clr_i[$clog2(N_OUT)-1:0]] <= 1'b0;
          if (clr_i == N_OUT-1) begin mem_we<=1'b0; state<=IDLE; end
          else clr_i <= clr_i + 1;
        end
        IDLE: begin
          done<=0; mem_we<=1'b0;
          if (start) begin
            j<=0; i<=0; acc<=0;
            in_addr<=0; w_addr<=0; b_addr<=0;
            mem_ra<=0;                  // request membrane[0]
            state<=MEMRD;
          end
        end
        // membrane[j] read issued; data (mem_rd) valid after this cycle
        MEMRD: begin
          in_addr<=0; w_addr<=j*N_IN; b_addr<=j;
          state<=S_ADDR;
        end
        S_ADDR: state<=S_DATA;
        S_DATA: begin
          if (in_spike) acc <= acc + w_data;
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
          beta_mem   = BETA * mem_rd;                  // membrane from BRAM
          leaked     = beta_mem >>> 10;
          mem_next32 = leaked + acc + b_data - (prev_spk[j] ? THRESH : 32'sd0);
          if (mem_next32 > SAT_MAX) mem_next32 = SAT_MAX;
          if (mem_next32 < SAT_MIN) mem_next32 = SAT_MIN;
          // write updated membrane back to BRAM
          mem_we<=1'b1; mem_wa<=j[$clog2(N_OUT)-1:0]; mem_wd<=mem_next32[15:0];
          if (mem_next32 >= THRESH) begin spike_out[j]<=1'b1; prev_spk[j]<=1'b1; end
          else                      begin spike_out[j]<=1'b0; prev_spk[j]<=1'b0; end
          state<=NEXTN;
        end
        NEXTN: begin
          mem_we<=1'b0;
          if (j == N_OUT-1) state<=FIN;
          else begin
            j      <= j + 1;
            i      <= 0;
            acc    <= 0;
            mem_ra <= j + 1;            // request next membrane
            state  <= MEMRD;
          end
        end
        FIN: begin done<=1; state<=IDLE; end
      endcase
    end
  end
endmodule
