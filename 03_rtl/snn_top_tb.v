`timescale 1ns/1ps
// Full-chain verification: conv1 -> fc1 -> fc2, against gen_ref_full.py.
module snn_top_tb;
  reg clk=0, rst=1, start=0;
  reg [131:0] conv1_in;
  wire [1:0] fc2_spike;
  wire signed [15:0] fc2_mem0, fc2_mem1;
  wire done;

  snn_top uut(.clk(clk), .rst(rst), .start(start),
              .conv1_in(conv1_in),
              .fc2_spike(fc2_spike), .fc2_mem0(fc2_mem0), .fc2_mem1(fc2_mem1),
              .done(done));

  always #5 clk=~clk;

  reg spk_in [0:2*132-1];
  reg exp_conv1 [0:2*528-1];
  reg exp_fc1   [0:2*256-1];
  reg exp_fc2sp [0:2*2-1];
  reg signed [15:0] exp_fc2mem [0:2*2-1];

  integer tstep, k, errors, c1cnt, f1cnt;

  initial begin
    $readmemh("conv1_w_q610.hex", uut.conv1_w_mem);
    $readmemh("conv1_b_q610.hex", uut.conv1_b_mem);
    $readmemh("fc1_w_q610.hex",   uut.fc1_w_mem);
    $readmemh("fc1_b_q610.hex",   uut.fc1_b_mem);
    $readmemh("fc2_w_q610.hex",   uut.fc2_w_mem);
    $readmemh("fc2_b_q610.hex",   uut.fc2_b_mem);

    $readmemb("full_conv1_spike_in.b",         spk_in);
    $readmemb("full_conv1_spikes_expected.txt", exp_conv1);
    $readmemb("full_fc1_spikes_expected.txt",   exp_fc1);
    $readmemb("full_fc2_spikes_expected.txt",   exp_fc2sp);
    $readmemh("full_fc2_mem_expected.hex",      exp_fc2mem);

    errors=0;
    repeat(3) @(negedge clk); rst=0;
    repeat(600) @(negedge clk);   // let conv1's + fc1's membrane-clear passes finish

    for (tstep=0; tstep<2; tstep=tstep+1) begin
      for (k=0;k<132;k=k+1) conv1_in[k] = spk_in[tstep*132+k];
      @(negedge clk); start=1;
      @(negedge clk); start=0;
      wait(done==1);
      @(negedge clk);

      // check conv1 internal spikes
      for (k=0;k<528;k=k+1)
        if (uut.conv1_spk[k] !== exp_conv1[tstep*528+k]) begin
          errors=errors+1;
          if (errors<=3) $display("  conv1 mismatch t=%0d k=%0d", tstep, k);
        end
      // check fc1 internal spikes
      for (k=0;k<256;k=k+1)
        if (uut.fc1_spk[k] !== exp_fc1[tstep*256+k]) begin
          errors=errors+1;
          if (errors<=3) $display("  fc1 mismatch t=%0d k=%0d", tstep, k);
        end
      // check fc2 (final) spikes + membranes
      if (fc2_spike[0] !== exp_fc2sp[tstep*2+0]) errors=errors+1;
      if (fc2_spike[1] !== exp_fc2sp[tstep*2+1]) errors=errors+1;
      if (fc2_mem0 !== exp_fc2mem[tstep*2+0]) errors=errors+1;
      if (fc2_mem1 !== exp_fc2mem[tstep*2+1]) errors=errors+1;

      c1cnt=0; for (k=0;k<528;k=k+1) c1cnt=c1cnt+uut.conv1_spk[k];
      f1cnt=0; for (k=0;k<256;k=k+1) f1cnt=f1cnt+uut.fc1_spk[k];
      $display("timestep %0d: conv1 spikes=%0d/528  fc1 spikes=%0d/256  fc2 spike=%b  fc2 mem=[%0d,%0d]  (expected mem=[%0d,%0d])",
        tstep, c1cnt, f1cnt, fc2_spike,
        fc2_mem0, fc2_mem1, exp_fc2mem[tstep*2+0], exp_fc2mem[tstep*2+1]);
    end

    if (errors==0) $display("RESULT: PASS - full conv1->fc1->fc2 chain bit-exact (spikes + membranes)");
    else           $display("RESULT: FAIL - %0d mismatches", errors);
    $finish;
  end
endmodule
