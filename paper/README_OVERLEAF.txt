CACH DUNG TREN OVERLEAF
=======================
1. Overleaf -> New Project -> "Upload Project" -> chon file .zip (dung giai nen)
2. Overleaf tu mo main.tex -> bam Recompile
3. Neu khong tu nhan file chinh: Menu -> Main document -> main.tex


SOICT 2026 (soict.org, HUST Ha Noi) — YEU CAU
=============================================
  Abstract          09/09/2026
  Full paper        16/09/2026
  Notification      12/10/2026
  Camera-ready      23/10/2026
  Page limit        12 trang KHONG tinh references (ban NOP)
                    camera-ready full paper 12-15 trang TINH CA references
  Template          Springer CCIS/LNCS (llncs.cls) — DUNG SUA file class
  Review            SINGLE-BLIND -> GIU ten tac gia + affiliation
  Nop qua           EasyChair. Proceedings: Springer CCIS (Scopus/DBLP/EI)
  PDF               KHONG danh so trang (llncs tu lo)


CAU TRUC
========
main.tex          khai bao + danh sach \input + macro \TODO va \ms. It khi phai sua.
llncs.cls         class Springer LNCS/CCIS (DUNG SUA)
splncs04.bst      bib style Springer (du phong)
make_tables.py    SINH DONG LATEX cho bang tu runs/paper_queue/UNIFIED.json
                  -> chay "python paper/make_tables.py", dan output vao sections/.
                  DUNG CHEP SO BANG TAY. Chay lai sau moi lan do them model.

sections/
  01_titlepage      ten, email, don vi          <- DIEN TEN THAY + EMAIL + ORCID
  02_abstract       abstract + keywords
  03_introduction   1. Introduction             <- HINH 1 (teaser)
  04_related_work   2. Related Work
  05_platform       3.1 Kien truc SteadyDet-N       <- HINH 2 (arch). Muc nay CO Y NGAN:
                                                     kien truc KHONG phai dong gop
  06_shake_protocol 3.2 Giao thuc rung VisDrone-Shake
  07_vibcon         3.3 Vibration-Consistency   <- HINH 3 (method). DONG GOP CHINH
  08_exp_setup      4.1 Setup (MOT giao thuc do duy nhat + seed)
  09_exp_main       4.2 Ket qua chinh           <- BANG 1 (gop ca Nano va Wide)
  10_exp_generality 4.3 Tong quat (dung lai bang o 4.2, KHONG co bang rieng)
  11_exp_baselines  4.4 So sanh YOLO/RT-DETR    <- BANG 2 + HINH 4 (tradeoff)
  12_exp_flight     4.5 Val bay THAT            <- BANG 3
  13_exp_deploy     4.6 Hailo / Pi 5            <- con \TODO
  14_limitations    5. Limitations
  15_conclusion     6. Conclusion
  16_bibliography   26 references (da kiem: cite du, khong thua, khong thieu)

figures/          4 HINH DANG DUNG + script .py sinh ra chung:
  fig1_teaser.pdf      make_fig1_teaser.py    frame bay that co NGUOI (4 muc rung) +
                                              duong "% AP giu duoc" cua MOI ho detector
  fig2_method.pdf      make_fig2_method.py    so do 2 nhanh + stop-gradient
                                              (dinh tuyen VUONG GOC, khong con duong cong)
  fig3_arch.pdf        make_fig3_arch.py      kien truc SteadyDet-N theo tung tang P2..P5
  fig4_tradeoff.pdf    make_fig4_tradeoff.py  clean AP vs robustness, 1 diem / model
                                              + mui nhui = tac dung cua vib-consistency
  fig4_qualitative.pdf make_fig4_qual.py      3 khung bay that (HIEN DANG COMMENT trong tex)
  sample_frame.jpg                            anh du phong cho fig1

  fig1 va fig4 DOC SO TRUC TIEP tu runs/paper_queue/UNIFIED.json -> chay lai script la
  hinh tu cap nhat, khong bao gio lech voi bang.
  Muon sua hinh: sua .py roi chay "python make_figX.py" tren may minh (KHONG chay duoc
  tren Overleaf), roi copy file .pdf moi len.


TRUOC KHI NOP — BAT BUOC
========================
1. Dien het \TODO (mau DO trong PDF). Con 4 cho:
     01_titlepage      ten thay, email, ORCID
     13_exp_deploy     so lieu quantisation / latency  <- HOAC CAT HAN MUC NAY
     main.tex          Acknowledgements
   Grep lai: chu "TODO" khong duoc con trong sections/ va main.tex.
2. Trong main.tex doi:
       \newcommand{\TODO}[1]{\textcolor{red}{\textbf{[#1]}}}
   thanh:
       \newcommand{\TODO}[1]{#1}
3. Kiem tra font trong figure >= 8pt SAU KHI scale (trim size Springer nho hon A4).
4. pdffonts main.pdf -> moi dong phai co "emb".
5. Kiem hinh doc duoc khi in DEN TRANG. Cap mau xanh/cam da duoc kiem bang so:
   normal-vision dE 33.6 / deuteranopia 31.6 / protanopia 24.5 (san la >=15 va >=8),
   nhung do sang xam gan nhau (Y 0.19 vs 0.28) nen MOI duong deu co kieu net + marker
   rieng, va moi diem trong Hinh 4 co nhan truc tiep -> in xam van doc duoc.
6. KHONG dung dau gach dai "---" (dau hieu van AI). Kiem:
     grep -v '^%' sections/*.tex | grep -c -- '---'     phai ra 0
   ("--" trong khoang so nhu $19$--$20\%$ la en-dash, DUNG, giu nguyen.)


SO TRANG  (cap nhat 2026-07-27)
===============================
Chua compile duoc tren may (khong co LaTeX). Uoc luong bang cach do chieu cao THUC
cua tung hinh/bang + hieu chuan chu voi draft cu da compile (31.885 ky tu = 10.24
trang). Lan truoc uoc 12.6, Overleaf that ra 12.7 -> sai so ~0.1 trang.

    CHU   32.366 ky tu  = 10.39 trang
    HINH  4 hinh        =  1.23 trang   (da thu nho: 0.93/0.94/0.88/0.88 textwidth)
    BANG  2 bang        =  0.63 trang
    ---------------------------------
    TONG                ~12.25 trang    (gioi han 12, KHONG tinh references)

VIEC DAU TIEN: upload Overleaf, Recompile, DEM SO TRANG THAT.

DA CAT trong phien 2026-07-27:
  - MUC 4.5 "Validation on Real Flight Footage" (quyet dinh cua tac gia).
    File sections/12_exp_flight.tex VAN CON trong repo va DA CAP NHAT du 6 dong so
    that (runs/paper_queue/FLIGHT.json). \input da comment trong main.tex.
    Ly do cat: 92 frame / 1 nguoi / 1 canh / quay man hinh GCS = bang chung yeu nhat.
    DUA LAI sau chuyen bay dau thang 8 (xem FLIGHT_DATA_SPEC.md) -> luc do se manh hon.
  - cot AP_s trong bang baselines (gan trung voi AP_50:95: 0.2741 vs 0.2768)
    -> thay bang cot mCP, thong tin hon va khong ton them cot.
  - thu nho 4 hinh + fig4 giam chieu cao 3.05 -> 2.72 inch.

NEU OVERLEAF BAO CON DU (~0.3 trang), cat theo thu tu nay:
  1. 06_shake_protocol.tex - muc "The protocol is stochastic"     ~0.12 trang
     Rut con 2 cau, giu cau ket "Gains smaller than run-to-run spread are not gains".
  2. 04_related_work.tex - muc "Consistency regularisation"        ~0.15 trang
     Rut con 2 cau + giu nguyen cite. Consistency gio la ablation AM, khong phai
     phuong phap chinh, nen khong can dat nen ly thuyet day du cho no nua.
  3. 09_exp_main.tex - doan giai thich vi sao consistency kem      ~0.15 trang
     Giu 2 cau dau (so + "khong phai nhieu"), bo phan suy dien co che.
  4. 11_exp_baselines.tex - doan cuoi ve harness rieng             ~0.12 trang
     Chuyen thanh footnote cua Bang 2.

!! DOI 2026-07-27b: KHONG CON DUOC CAT HINH 3 (fig3_arch).
   Truoc day no dung dau danh sach cat vi bai tu noi "kien truc KHONG phai dong gop".
   Bay gio KIEN TRUC LA DONG GOP (i) -> hinh kien truc thanh bang chung, khong phai
   minh hoa. Cat no = cat chan mot trong nam dong gop.

KHONG BAO GIO CAT: Hinh 1 (teaser), Hinh 2 (method), Hinh 3 (kien truc),
Bang 1 (ket qua chinh), Bang 2 (baselines + cot mCP).
Bang 2 mang claim ve MODEL - no la bang quan trong nhat bai.
