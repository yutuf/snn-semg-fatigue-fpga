`timescale 1ns/1ps
module snn_fc2_output_tb;
  localparam N_IN=256, N_OUT=2;

  reg clk=0, rst=1, start=0;
  wire in_spike;
  wire [$clog2(N_IN)-1:0] in_addr;
  wire [$clog2(N_IN*N_OUT)-1:0] w_addr;
  wire [$clog2(N_OUT)-1:0] b_addr;
  wire [N_OUT-1:0] spike_out;
  wire signed [16*N_OUT-1:0] mem_out_flat;
  wire done;

  reg signed [15:0] w_mem [0:N_IN*N_OUT-1];
  reg signed [15:0] b_mem [0:N_OUT-1];
  reg               spk_in [0:2*N_IN-1];
  reg signed [15:0] w_data_r, b_data_r;
  reg               in_spike_r;
  integer tstep;

  always @(posedge clk) begin
    w_data_r  <= w_mem[w_addr];
    b_data_r  <= b_mem[b_addr];
    in_spike_r<= spk_in[tstep*N_IN + in_addr];
  end

  snn_fc2_output #(.N_IN(N_IN), .N_OUT(N_OUT), .BETA(16'sd907), .THRESH(16'sd307))
    uut(.clk(clk), .rst(rst), .start(start),
        .in_spike(in_spike_r),
        .in_addr(in_addr), .w_addr(w_addr), .w_data(w_data_r),
        .b_addr(b_addr), .b_data(b_data_r),
        .spike_out(spike_out), .mem_out_flat(mem_out_flat), .done(done));

  assign in_spike = in_spike_r;
  always #5 clk=~clk;

  reg exp_spk [0:2*N_OUT-1];
  reg signed [15:0] exp_mem [0:2*N_OUT-1];
  integer errors, k, got;

  initial begin
    $readmemh("fc2_w_q610.hex", w_mem);
    $readmemh("fc2_b_q610.hex", b_mem);
    $readmemb("fc2_spike_in.b", spk_in);
    $readmemb("fc2_spikes_expected.txt", exp_spk);
    $readmemh("fc2_mem_expected.hex", exp_mem);

    errors=0; tstep=0;
    repeat(3) @(negedge clk); rst=0;

    for (tstep=0; tstep<2; tstep=tstep+1) begin
      @(negedge clk); start=1;
      @(negedge clk); start=0;
      wait(done==1);
      @(negedge clk);
      for (k=0;k<N_OUT;k=k+1) begin
        if (spike_out[k] !== exp_spk[tstep*N_OUT+k]) errors=errors+1;
        if ($signed(mem_out_flat[16*k +: 16]) !== exp_mem[tstep*N_OUT+k]) errors=errors+1;
      end
      $display("timestep %0d: spikes=%b  mem=[%0d,%0d]  expected mem=[%0d,%0d]",
        tstep, spike_out, $signed(mem_out_flat[15:0]), $signed(mem_out_flat[31:16]),
        exp_mem[tstep*N_OUT+0], exp_mem[tstep*N_OUT+1]);
    end

    if (errors==0) $display("RESULT: PASS - fc2 output layer bit-exact (spikes AND membranes)");
    else           $display("RESULT: FAIL - %0d mismatches", errors);
    $finish;
  end
endmodule
