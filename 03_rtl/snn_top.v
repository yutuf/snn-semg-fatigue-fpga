// ============================================================
// snn_top.v
// Full network, one timestep: conv1(4->16,k5,pad2)+LIF1 -> fc1(528->256)+LIF2
//                              -> fc2(256->2, output, membrane exposed)+LIF3
//
// Sequences the three verified cores: conv1's spike_out feeds fc1's
// in_spike stream (combinationally indexed), fc1's spike_out feeds fc2's.
// Weight/bias memories are internal (loaded via $readmemh here for
// sim/synthesis testing; on real hardware these map to SPRAM/EBR banks -
// see report for the separately-established memory budget).
// ============================================================
module snn_top (
  input  wire clk,
  input  wire rst,
  input  wire start,                 // begin processing one timestep
  input  wire [131:0] conv1_in,       // this timestep's 132 input spikes (4ch x 33)
  output wire [1:0]   fc2_spike,
  output wire signed [15:0] fc2_mem0,
  output wire signed [15:0] fc2_mem1,
  output reg   done
);
  // ---- conv1 ----
  wire conv1_done;
  wire [527:0] conv1_spk;
  wire [7:0]  conv1_in_addr;          // clog2(132)=8
  wire [8:0]  conv1_w_addr;           // clog2(320)=9
  wire [3:0]  conv1_b_addr;           // clog2(16)=4
  reg  conv1_start;

  reg signed [15:0] conv1_w_mem [0:319];
  reg signed [15:0] conv1_b_mem [0:15];
  initial begin
    $readmemh("conv1_w_q610.hex", conv1_w_mem);
    $readmemh("conv1_b_q610.hex", conv1_b_mem);
  end
  wire signed [15:0] conv1_w_data = conv1_w_mem[conv1_w_addr];
  wire signed [15:0] conv1_b_data = conv1_b_mem[conv1_b_addr];
  wire conv1_in_spike = conv1_in[conv1_in_addr];

  snn_conv1_core_bram #(.BETA(16'sd841), .THRESH(16'sd307)) u_conv1 (
    .clk(clk), .rst(rst), .start(conv1_start),
    .in_spike(conv1_in_spike), .in_addr(conv1_in_addr),
    .w_addr(conv1_w_addr), .w_data(conv1_w_data),
    .b_addr(conv1_b_addr), .b_data(conv1_b_data),
    .spike_out(conv1_spk), .done(conv1_done)
  );

  // ---- fc1 (reuses the generic, already-verified BRAM FC engine) ----
  wire fc1_done;
  wire [255:0] fc1_spk;
  wire [9:0]  fc1_in_addr;            // clog2(528)=10 (512<528, needs 10 bits)
  wire [17:0] fc1_w_addr;             // AW=18
  wire [7:0]  fc1_b_addr;             // clog2(256)=8
  reg  fc1_start;

  reg signed [15:0] fc1_w_mem [0:528*256-1];   // 135168 entries
  reg signed [15:0] fc1_b_mem [0:255];
  initial begin
    $readmemh("fc1_w_q610.hex", fc1_w_mem);
    $readmemh("fc1_b_q610.hex", fc1_b_mem);
  end
  wire signed [15:0] fc1_w_data = fc1_w_mem[fc1_w_addr];
  wire signed [15:0] fc1_b_data = fc1_b_mem[fc1_b_addr];
  wire fc1_in_spike = conv1_spk[fc1_in_addr];

  snn_fc_core_bram #(.N_IN(528), .N_OUT(256), .AW(18),
                     .BETA(16'sd933), .THRESH(16'sd307)) u_fc1 (
    .clk(clk), .rst(rst), .start(fc1_start),
    .in_spike(fc1_in_spike), .in_addr(fc1_in_addr),
    .w_addr(fc1_w_addr), .w_data(fc1_w_data),
    .b_addr(fc1_b_addr), .b_data(fc1_b_data),
    .spike_out(fc1_spk), .done(fc1_done)
  );

  // ---- fc2 (output layer, exposes membrane) ----
  wire fc2_done;
  wire [1:0] fc2_spk;
  wire [7:0] fc2_in_addr;             // clog2(256)=8
  wire [8:0] fc2_w_addr;              // clog2(512)=9
  wire       fc2_b_addr;              // clog2(2)=1
  wire signed [31:0] fc2_mem_flat;
  reg  fc2_start;

  reg signed [15:0] fc2_w_mem [0:511];
  reg signed [15:0] fc2_b_mem [0:1];
  initial begin
    $readmemh("fc2_w_q610.hex", fc2_w_mem);
    $readmemh("fc2_b_q610.hex", fc2_b_mem);
  end
  wire signed [15:0] fc2_w_data = fc2_w_mem[fc2_w_addr];
  wire signed [15:0] fc2_b_data = fc2_b_mem[fc2_b_addr];
  wire fc2_in_spike = fc1_spk[fc2_in_addr];

  snn_fc2_output #(.N_IN(256), .N_OUT(2),
                   .BETA(16'sd907), .THRESH(16'sd307)) u_fc2 (
    .clk(clk), .rst(rst), .start(fc2_start),
    .in_spike(fc2_in_spike), .in_addr(fc2_in_addr),
    .w_addr(fc2_w_addr), .w_data(fc2_w_data),
    .b_addr(fc2_b_addr), .b_data(fc2_b_data),
    .spike_out(fc2_spk), .mem_out_flat(fc2_mem_flat), .done(fc2_done)
  );

  assign fc2_spike = fc2_spk;
  assign fc2_mem0  = fc2_mem_flat[15:0];
  assign fc2_mem1  = fc2_mem_flat[31:16];

  // ---- top sequencer: conv1 -> fc1 -> fc2, one timestep per `start` pulse ----
  localparam T_IDLE=3'd0, T_CONV1=3'd1, T_WAIT_C=3'd2,
             T_FC1=3'd3, T_WAIT_F1=3'd4, T_FC2=3'd5, T_WAIT_F2=3'd6, T_DONE=3'd7;
  reg [2:0] tstate;

  always @(posedge clk) begin
    if (rst) begin
      tstate<=T_IDLE; conv1_start<=0; fc1_start<=0; fc2_start<=0; done<=0;
    end else begin
      conv1_start<=0; fc1_start<=0; fc2_start<=0;
      case (tstate)
        T_IDLE:    begin done<=0; if (start) begin conv1_start<=1; tstate<=T_WAIT_C; end end
        T_WAIT_C:  if (conv1_done) tstate<=T_FC1;
        T_FC1:     begin fc1_start<=1; tstate<=T_WAIT_F1; end
        T_WAIT_F1: if (fc1_done) tstate<=T_FC2;
        T_FC2:     begin fc2_start<=1; tstate<=T_WAIT_F2; end
        T_WAIT_F2: if (fc2_done) tstate<=T_DONE;
        T_DONE:    begin done<=1; tstate<=T_IDLE; end
        default:   tstate<=T_IDLE;
      endcase
    end
  end
endmodule
