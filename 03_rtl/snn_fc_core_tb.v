`timescale 1ns/1ps
// Testbench: verify snn_fc_core (528->256, external weight/bias mem) against
// the numpy Q6.10 reference. Behavioral weight/bias ROMs load real fc1 hex.
module snn_fc_core_tb;
  localparam N_IN=528, N_OUT=256, AW=18;

  reg clk=0, rst=1, start=0;
  wire in_spike;
  wire [$clog2(N_IN)-1:0] in_addr;
  wire [AW-1:0]           w_addr;
  wire [$clog2(N_OUT)-1:0] b_addr;
  wire [N_OUT-1:0]        spike_out;
  wire done;

  // external memories (behavioral; SPRAM on real device)
  reg signed [15:0] w_mem [0:N_IN*N_OUT-1];
  reg signed [15:0] b_mem [0:N_OUT-1];
  reg               spk_in [0:2*N_IN-1];   // 2 timesteps of input spikes
  reg signed [15:0] w_data_r, b_data_r;
  reg               in_spike_r;

  integer tstep;

  // registered memory reads (1-cycle latency, matches SPRAM)
  always @(posedge clk) begin
    w_data_r  <= w_mem[w_addr];
    b_data_r  <= b_mem[b_addr];
    in_spike_r<= spk_in[tstep*N_IN + in_addr];
  end

  snn_fc_core #(.N_IN(N_IN), .N_OUT(N_OUT), .AW(AW),
                .BETA(16'sd933), .THRESH(16'sd307))
    uut(.clk(clk), .rst(rst), .start(start),
        .in_spike(in_spike_r),
        .in_addr(in_addr), .w_addr(w_addr), .w_data(w_data_r),
        .b_addr(b_addr), .b_data(b_data_r),
        .spike_out(spike_out), .done(done));

  assign in_spike = in_spike_r;

  always #5 clk=~clk;

  // expected data
  reg exp_spk [0:2*N_OUT-1];
  integer errors, k, got;
  reg [8*64:1] junk;

  initial begin
    $readmemh("fc1_w_q610.hex", w_mem);
    $readmemh("fc1_b_q610.hex", b_mem);
    $readmemb("spike_in.b", spk_in);
    // spikes_expected.txt is 0/1 per line -> load as memb
    $readmemb("spikes_expected.txt", exp_spk);

    errors=0;
    repeat(3) @(negedge clk); rst=0;

    for (tstep=0; tstep<2; tstep=tstep+1) begin
      @(negedge clk); start=1;
      @(negedge clk); start=0;
      wait(done==1);
      @(negedge clk);
      got=0;
      for (k=0;k<N_OUT;k=k+1) begin
        if (spike_out[k] !== exp_spk[tstep*N_OUT+k]) errors=errors+1;
        got = got + spike_out[k];
      end
      $display("timestep %0d: hw spikes=%0d  (compared %0d neurons)", tstep, got, N_OUT);
    end

    if (errors==0) $display("RESULT: PASS - bit-exact match on all %0d neuron-spikes over 2 timesteps", 2*N_OUT);
    else           $display("RESULT: FAIL - %0d mismatches", errors);
    $finish;
  end
endmodule
