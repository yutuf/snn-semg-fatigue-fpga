// ============================================================
// snn_fc_layer.v
// Zaman-paylasimli (time-multiplexed) tam-bagli (FC) + LIF katman motoru.
// TEK paylasimli hesap birimi tum norönlari sirayla isler.
// Norön membran durumlari mem_arr[] icinde tutulur.
// Spike-suruculu: sadece giris spike'i 1 oldugunda agirlik toplanir (AC).
//
// Q6.10 sabit-nokta, 16-bit signed.
// Agirliklar WFILE'dan (hex) $readmemh ile yuklenir: N_OUT*N_IN eleman,
//   sira: neuron0'in tum girisleri, sonra neuron1... (j*N_IN + i)
// ============================================================
module snn_fc_layer #(
  parameter N_IN   = 4,
  parameter N_OUT  = 2,
  parameter signed [15:0] BETA   = 16'sd870,   // 0.85 Q6.10 (varsayilan; ust modul override eder)
  parameter signed [15:0] THRESH = 16'sd307,   // 0.30 Q6.10
  parameter WFILE  = "weights.hex"
)(
  input  wire clk,
  input  wire rst,           // membranlari sifirla (yeni dizi/sekans basi)
  input  wire start,         // bu zaman-adimini isle
  input  wire [N_IN-1:0] spike_in,   // giris spike vektoru (binary)
  output reg  [N_OUT-1:0] spike_out,  // cikis spike vektoru
  output reg  done
);
  localparam signed [15:0] SAT_MAX =  16'sd32767;
  localparam signed [15:0] SAT_MIN = -16'sd32768;

  // Agirlik ve membran bellekleri
  reg signed [15:0] w_mem [0:N_OUT*N_IN-1];
  reg signed [15:0] mem_arr [0:N_OUT-1];
  reg               prev_spk [0:N_OUT-1];

  initial $readmemh(WFILE, w_mem);

  // FSM
  localparam IDLE=2'd0, ACC=2'd1, FIRE=2'd2, FIN=2'd3;
  reg [1:0] state;
  integer j, i;                       // j: cikis norön, i: giris indeksi
  reg signed [31:0] acc;              // genis akumulator (tasma korumasi)

  // Membranlari reset'te sifirla
  integer r;
  always @(posedge clk) begin
    if (rst) begin
      for (r=0; r<N_OUT; r=r+1) begin mem_arr[r] <= 0; prev_spk[r] <= 1'b0; end
      spike_out <= 0; done <= 0; state <= IDLE; j <= 0; i <= 0; acc <= 0;
    end else begin
      case (state)
        IDLE: begin
          done <= 0;
          if (start) begin j <= 0; i <= 0; acc <= 0; state <= ACC; end
        end
        ACC: begin
          // giris i icin: spike varsa agirligi ekle (AC islemi)
          if (spike_in[i]) acc <= acc + w_mem[j*N_IN + i];
          if (i == N_IN-1) state <= FIRE;
          else i <= i + 1;
        end
        FIRE: begin
          // LIF: mem = beta*mem + acc - (prev_spike? thresh:0)
          // leaked = (BETA * mem_arr[j]) >>> 10
          begin : lif_calc
            reg signed [31:0] beta_mem, mem_next32;
            reg signed [15:0] leaked;
            beta_mem = BETA * mem_arr[j];
            leaked   = beta_mem >>> 10;
            mem_next32 = leaked + acc - (prev_spk[j] ? THRESH : 16'sd0);
            // saturasyon -> 16 bit
            if (mem_next32 >  SAT_MAX) mem_next32 = SAT_MAX;
            if (mem_next32 <  SAT_MIN) mem_next32 = SAT_MIN;
            mem_arr[j] <= mem_next32[15:0];
            if (mem_next32 >= THRESH) begin
              spike_out[j] <= 1'b1; prev_spk[j] <= 1'b1;
            end else begin
              spike_out[j] <= 1'b0; prev_spk[j] <= 1'b0;
            end
          end
          // sonraki norön
          if (j == N_OUT-1) state <= FIN;
          else begin j <= j + 1; i <= 0; acc <= 0; state <= ACC; end
        end
        FIN: begin done <= 1; state <= IDLE; end
      endcase
    end
  end
endmodule
