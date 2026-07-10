// ============================================================
// snn_conv1_core_bram.v
// Time-multiplexed Conv1d(4->16, k=5, pad=2) + LIF1 engine.
//
// Unlike the FC layers, conv1 has WEIGHT SHARING: the same 320 weights
// (16 out_ch * 4 in_ch * 5 taps) + 16 biases are reused across all 33
// spatial (frequency-bin) positions. We therefore do NOT expand/duplicate
// the weight table (that would need ~1 Mbit and blow the memory budget) -
// the real 336-parameter weight set is read from external memory exactly
// as PyTorch stores it: flat index = c_out*20 + c_in*5 + k.
//
// Input : 132 spike bits (4 channels x 33 positions), addressed
//         in_addr = c_in*33 + pos   (matches Conv1d input layout (C,L))
// Output: 528 spike bits (16 channels x 33 positions), addressed
//         j = c_out*33 + p          (matches PyTorch .view(batch,-1) of (16,33))
//
// Zero-padding (pad=2): taps with pos+k-2 outside [0,32] contribute 0.
// Membranes (528) held in inferred BRAM, per the lesson learned on fc1
// (528 > 256, so flip-flop storage would be even worse than the fc1 case).
//
// Q6.10 fixed-point, 16-bit signed datapath.
// ============================================================
module snn_conv1_core_bram #(
  parameter N_CIN  = 4,
  parameter N_COUT = 16,
  parameter K      = 5,
  parameter PAD    = 2,
  parameter N_POS  = 33,
  parameter N_OUT  = N_COUT*N_POS,          // 528
  parameter signed [15:0] BETA   = 16'sd841,  // lif1 0.82164 Q6.10
  parameter signed [15:0] THRESH = 16'sd307   // 0.30   Q6.10
)(
  input  wire clk,
  input  wire rst,
  input  wire start,
  input  wire in_spike,                          // input[in_addr] spike bit
  output reg  [$clog2(N_CIN*N_POS)-1:0] in_addr,  // 0..131
  output reg  [$clog2(N_COUT*N_CIN*K)-1:0] w_addr,// 0..319 (c_out*20+c_in*5+k)
  input  wire signed [15:0]       w_data,
  output reg  [$clog2(N_COUT)-1:0] b_addr,        // 0..15
  input  wire signed [15:0]       b_data,
  output reg  [N_OUT-1:0]         spike_out,      // 528 bits, j=c_out*33+p
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

  reg prev_spk [0:N_OUT-1];   // cheap: only N_OUT bits

  localparam CLR=4'd0, IDLE=4'd1, MEMRD=4'd2, TAP_ADDR=4'd3, TAP_WAIT=4'd4,
             TAP_ACC=4'd5, BIAS_ADDR=4'd6, BIAS_WAIT=4'd7, FIRE=4'd8,
             NEXTP=4'd9, FIN=4'd10;
  reg [3:0] state;

  reg [$clog2(N_COUT)-1:0] c_out;              // 0..15
  reg [$clog2(N_POS)-1:0]  p;                   // 0..32
  reg [$clog2(N_OUT)-1:0]  j;                   // c_out*33+p, incremented alongside
  reg [$clog2(N_CIN)-1:0]  c_in;                // 0..3
  reg [2:0]                k;                   // 0..4  (K=5)
  reg [$clog2(N_COUT*N_CIN*K)-1:0] w_base_cout; // c_out*20, incremental (+20)
  reg [$clog2(N_CIN*N_POS)-1:0]    cin_base;    // c_in*33, incremental (+33)
  reg signed [31:0] acc;

  // Combinational (not registered): always reflects current p,k, valid
  // identically across TAP_ADDR/TAP_WAIT/TAP_ACC for a given tap (p,k
  // only change when NEXTP/TAP_ACC advances to the next tap).
  wire signed [8:0] pos_in = $signed({3'b0,p}) + $signed({6'b0,k}) - PAD;
  wire pos_valid = (pos_in >= 0) && (pos_in < N_POS);

  integer clr_i;

  always @(posedge clk) begin
    if (rst) begin
      spike_out<=0; done<=0; c_out<=0; p<=0; j<=0; c_in<=0; k<=0;
      w_base_cout<=0; cin_base<=0; acc<=0;
      in_addr<=0; w_addr<=0; b_addr<=0;
      clr_i<=0; mem_we<=1'b1; mem_wa<=0; mem_wd<=16'sd0;
      state<=CLR;
    end else begin
      case (state)
        CLR: begin
          mem_we<=1'b1; mem_wa<=clr_i[$clog2(N_OUT)-1:0]; mem_wd<=16'sd0;
          prev_spk[clr_i[$clog2(N_OUT)-1:0]] <= 1'b0;
          if (clr_i == N_OUT-1) begin mem_we<=1'b0; state<=IDLE; end
          else clr_i <= clr_i + 1;
        end
        IDLE: begin
          done<=0; mem_we<=1'b0;
          if (start) begin
            c_out<=0; p<=0; j<=0; w_base_cout<=0;
            state<=MEMRD;
          end
        end
        // membrane[j] read issued; mem_rd valid one cycle later
        MEMRD: begin
          mem_ra<=j;
          c_in<=0; cin_base<=0; k<=0; acc<=32'sd0;
          state<=TAP_ADDR;
        end
        // pos_in/pos_valid are combinational (see above); just act on them
        TAP_ADDR: begin
          if (pos_valid) begin
            in_addr <= cin_base + pos_in[$clog2(N_CIN*N_POS)-1:0];
            w_addr  <= w_base_cout + {c_in,2'b0} + c_in + k; // c_in*5 = c_in*4+c_in
            state   <= TAP_WAIT;
          end else begin
            state   <= TAP_ACC;   // zero-pad: contributes nothing, skip read
          end
        end
        TAP_WAIT: state <= TAP_ACC;   // 1 cycle for in_spike/w_data to settle
        TAP_ACC: begin
          if (pos_valid) begin
            if (in_spike) acc <= acc + w_data;
          end
          // advance (c_in,k)
          if (k == K-1) begin
            k <= 0;
            if (c_in == N_CIN-1) begin
              state <= BIAS_ADDR;
            end else begin
              c_in     <= c_in + 1;
              cin_base <= cin_base + N_POS;
              state    <= TAP_ADDR;
            end
          end else begin
            k     <= k + 1;
            state <= TAP_ADDR;
          end
        end
        BIAS_ADDR: begin b_addr <= c_out; state <= BIAS_WAIT; end
        BIAS_WAIT: state <= FIRE;
        FIRE: begin : lif_calc
          reg signed [31:0] beta_mem, mem_next32;
          reg signed [15:0] leaked;
          beta_mem   = BETA * mem_rd;
          leaked     = beta_mem >>> 10;
          mem_next32 = leaked + acc + b_data - (prev_spk[j] ? THRESH : 32'sd0);
          if (mem_next32 > SAT_MAX) mem_next32 = SAT_MAX;
          if (mem_next32 < SAT_MIN) mem_next32 = SAT_MIN;
          mem_we<=1'b1; mem_wa<=j; mem_wd<=mem_next32[15:0];
          if (mem_next32 >= THRESH) begin spike_out[j]<=1'b1; prev_spk[j]<=1'b1; end
          else                      begin spike_out[j]<=1'b0; prev_spk[j]<=1'b0; end
          state<=NEXTP;
        end
        NEXTP: begin
          mem_we<=1'b0;
          if (j == N_OUT-1) begin
            state<=FIN;
          end else begin
            j <= j + 1;
            if (p == N_POS-1) begin
              p <= 0; c_out <= c_out + 1; w_base_cout <= w_base_cout + (N_CIN*K);
            end else begin
              p <= p + 1;
            end
            state <= MEMRD;
          end
        end
        FIN: begin done<=1; state<=IDLE; end
      endcase
    end
  end
endmodule
