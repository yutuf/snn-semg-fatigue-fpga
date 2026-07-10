`timescale 1ns/1ps
// Verify snn_conv1_core_bram (weight-shared conv + BRAM membranes)
// against the numpy Q6.10 reference (gen_ref_conv1.py).
module snn_conv1_core_bram_tb;
  localparam N_CIN=4, N_COUT=16, K=5, N_POS=33;
  localparam N_IN=N_CIN*N_POS, N_OUT=N_COUT*N_POS;     // 132, 528
  localparam WAW=9;   // clog2(16*4*5=320)

  reg clk=0, rst=1, start=0;
  wire in_spike;
  wire [$clog2(N_IN)-1:0] in_addr;
  wire [WAW-1:0]          w_addr;
  wire [$clog2(N_COUT)-1:0] b_addr;
  wire [N_OUT-1:0]        spike_out;
  wire done;

  reg signed [15:0] w_mem [0:N_COUT*N_CIN*K-1];   // 320
  reg signed [15:0] b_mem [0:N_COUT-1];            // 16
  reg               spk_in [0:2*N_IN-1];
  reg signed [15:0] w_data_r, b_data_r;
  reg               in_spike_r;
  integer tstep;

  always @(posedge clk) begin
    w_data_r  <= w_mem[w_addr];
    b_data_r  <= b_mem[b_addr];
    in_spike_r<= spk_in[tstep*N_IN + in_addr];
  end

  snn_conv1_core_bram #(.N_CIN(N_CIN), .N_COUT(N_COUT), .K(K), .PAD(2), .N_POS(N_POS),
                        .BETA(16'sd841), .THRESH(16'sd307))
    uut(.clk(clk), .rst(rst), .start(start),
        .in_spike(in_spike_r),
        .in_addr(in_addr), .w_addr(w_addr), .w_data(w_data_r),
        .b_addr(b_addr), .b_data(b_data_r),
        .spike_out(spike_out), .done(done));

  assign in_spike = in_spike_r;
  always #5 clk=~clk;

  reg exp_spk [0:2*N_OUT-1];
  integer errors, k2, got;

  initial begin
    $readmemh("conv1_w_q610.hex", w_mem);
    $readmemh("conv1_b_q610.hex", b_mem);
    $readmemb("conv1_spike_in.b", spk_in);
    $readmemb("conv1_spikes_expected.txt", exp_spk);

    errors=0; tstep=0;
    repeat(3) @(negedge clk); rst=0;
    repeat(600) @(negedge clk);      // allow membrane-clear pass (528 entries) to finish

    for (tstep=0; tstep<2; tstep=tstep+1) begin
      @(negedge clk); start=1;
      @(negedge clk); start=0;
      wait(done==1);
      @(negedge clk);
      got=0;
      for (k2=0;k2<N_OUT;k2=k2+1) begin
        if (spike_out[k2] !== exp_spk[tstep*N_OUT+k2]) errors=errors+1;
        got = got + spike_out[k2];
      end
      $display("timestep %0d: hw spikes=%0d", tstep, got);
    end

    if (errors==0) $display("RESULT: PASS - conv1 core bit-exact on all %0d neuron-spikes", 2*N_OUT);
    else           $display("RESULT: FAIL - %0d mismatches", errors);
    $finish;
  end
endmodule
