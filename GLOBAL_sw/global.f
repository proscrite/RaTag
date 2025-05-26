      PROGRAM GLOBAL
C     MF version 01/97
************************************************************************
C     f77 global.f -L/usr/local/cern/pro/lib -lkernlib -lmathlib -o global
************************************************************************
C HOUSE KEEPING REMARKS
C
C     Version 08/96: includes EN adjustment, changes in Output,
C                    new I_WRI fro WRITE frequency
C             09/96: contains D_EQ criterium to check equilibrium
C                    thickness
C                    DELT calculated from LIS or MIS, if too big
C                    devided by 10
C                    DELT no longer in MENUE
C                    Energy criterium for new x-section calculation
C                        log10(en)**2
C                    no solid-state factors for gas targets
C             10/96: J=max+5 (before +2)
C                    all variables defined
C                    new subroutine uppercase
C                    "cleaned" version
C                    modified SUBs PILO, PILO10, PRSO
C             11/96: modified output: Eout, D_eq in colums
C                    int-step / 10
C             01/97: debugged from test persons, mod. out-freq.,
C                    correct A,Z,Symbol, minor buggs corrected 
C             02/97: I_WRITE_FAC added, but should probably be removed again
C                    DELT removed from MENUE subroutine
C                    charge changed to Z for the loop question
C                    "Target step size too large" is written also on file
C***********************************************************************
C     GLOBAL                                                           *
C     A program to calculate and print out charge distributions        *
C     of relativistic projectiles traversing solid or gaseous targets. *
C     A short description how to handle the program is given in the    *
C     file global.readme. The physics basis of the program is described*
C     in W.E. Meyerhof et al., Nucl. Inst. Meth. B (to be published)   *
C                                                                      *
C     W.E. Meyerhof, Stanford University, 1985 - 1996                  *
C     E-mail: FE.WEM@FORSYTHE.STANFORD.EDU                             *
C     Bertram Blank, CEN Bordeaux-Gradignan, 1993 - 1996               *
C     E-mail: blank@cenbg.in2p3.fr                                     *
C***********************************************************************
C     Notation: ZF,AF:  proj. Z,A;  ZT,AT:  target Z,A                 *
C               X: calculated charge fractions                         *
C               DX: change of charge-state fraction in integration step*
C               Exx: incident-energy-related variables (almost always!)*
C               Dxx: target-thickness-related variables                *
C               Qxx: charge-states-related variables                   *
C               TXx: integration step sizes                            *
C               Uxx: ionisation potentials for different shells        *
C***********************************************************************
      REAL*4 DELDT
      REAL*4 ZFMIN, DELZF, ZTMIN, DELZT, EMIN, DELE, QMIN, DELQ, DTMIN
      REAL*4 PRODAT(7,96), TARDAT(3,97)
      REAL*4 NT, DE, EOUT, D_EQ
      REAL*4 U1S, U1SS, U2S, U2P, U3S, U3D
      REAL*4 TX, TX0, TX1, TX2, TX3, TX4, TX5
      REAL*4 X(0:30), DX(0:30), SA(0:28), FA(0:28)
      REAL*4 EA, SX, ZSQ, AUX, ZF0, ZT0, AF0, AT0, QIN0, RG, DTARG
      REAL*4 DTARGET0, EN, EDIFF, ENKEEP, ENMAX
C
      INTEGER*4 IP, IT, J, IRC, I, I_EN, IMAXLOOP, I_EQUI, IR, I_GAS
      INTEGER*4 LOOP/0/, I_WRITE_FAC
C MFA
      INTEGER*4 ID, IY
C MFE
C PCA
C     INTEGER*4 IYEAR, IMONTH, IDAY, IHOUR, IMINUTE, ISECOND, IHUND
C PCE
C
      CHARACTER*1 CYES /'Y'/
      CHARACTER*2 C
      CHARACTER*7 CQST(0:9)
      CHARACTER*8 CI
C **********************************************************************
      REAL*4        A(0:172)
      COMMON / AA / A
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
      REAL*4          SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,   MIS
      COMMON /BINPOL/ SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,   MIS
C **********************************************************************
      INTEGER*4 ISL(6)
      COMMON /SLATE/ ISL
C **********************************************************************
      CHARACTER*3 CPRO, CTAR
      CHARACTER*5 CVERS
      REAL*4 AF, ZF, AT, ZT, DTARGET, EN0, DELT
      INTEGER*4 I_CHAR, I_OUTP, I_BRAN, IMAX, JQST(0:9), I_LOOP, QIN
      INTEGER*4 I_WRI
      COMMON /GLOB/
     #       AF, ZF, AT, ZT, DTARGET, EN0, DELT, I_WRI,
     #       I_CHAR, I_OUTP, I_BRAN, IMAX, JQST, I_LOOP, QIN,
     #       CPRO, CTAR, CVERS
C **********************************************************************
C     Proj. ioniz. potentials in eV
C     from T.A. Carlson et al., At Data 2, (1970) 63ff
C                     Z      1S1     1S2    2S1    2P4    3S1    3D6
      DATA PRODAT  /  1.,   13.6,     1.,    1.,    1.,    1.,    1.,
     #                2.,  54.33,  24.98,    1.,    1.,    1.,    1.,
     #                3.,   122.,  74.45, 5.343,    1.,    1.,    1.,
     #                4.,   215.,   149., 18.69,    1.,    1.,    1.,
     #                5.,   333.,   249., 39.58,    1.,    1.,    1.,
     #                6.,   476.,   374., 67.59,    1.,    1.,    1.,
     #                7.,   643.,   524.,  103.,    1.,    1.,    1.,
     #                8.,   836.,   699.,  146.,    1.,    1.,    1.,
     #                9.,  1054.,   898.,  195.,    1.,    1.,    1.,
     #               10.,  1296.,  1123.,  252., 23.08,    1.,    1.,
     #               11.,  1575.,  1384.,  322., 47.69, 4.962,    1.,
     #               12.,  1880.,  1671.,  396., 78.71, 15.27,    1.,
     #               13.,  2211.,  1984.,  471.,  116., 29.15,    1.,
     #               14.,  2569.,  2324.,  552.,  160., 46.65,    1.,
     #               15.,  2953.,  2690.,  639.,  210., 67.88,    1.,
     #               16.,  3364.,  3082.,  733.,  266., 92.62,    1.,
     #               17.,  3801.,  3501.,  833.,  328.,  121.,    1.,
     #               18.,  4264.,  3947.,  939.,  396.,  152.,    1.,
     #               19.,  4759.,  4423., 1055.,  474.,  188.,    1.,
     #               20.,  5280.,  4926., 1178.,  558.,  226.,    1.,
     #               21.,  5827.,  5454., 1303.,  645.,  265.,    1.,
     #               22.,  6401.,  6009., 1435.,  738.,  307.,    1.,
     #               23.,  7002.,  6592., 1573.,  836.,  352.,    1.,
     #               24.,  7624.,  7195., 1711.,  934.,  394.,    1.,
     #               25.,  8285.,  7838., 1868., 1051.,  452.,    1.,
     #               26.,  8969.,  8503., 2025., 1168.,  506.,    1.,
     #               27.,  9679.,  9195., 2189., 1291.,  563.,    1.,
     #               28., 10420.,  9915., 2358., 1419.,  624.,    1.,
     #               29., 11170., 10650., 2523., 1542.,  677.,  21.2,
     #               30., 11980., 11440., 2717., 1694.,  754.,  39.6,
     #               31., 12810., 12250., 2918., 1852.,  835.,  61.6,
     #               32., 13670., 13090., 3125., 2015.,  917.,   87.,
     #               33., 14560., 13970., 3390., 2184., 1002.,  116.,
     #               34., 15480., 14860., 3559., 2359., 1089.,  147.,
     #               35., 16430., 15790., 3787., 2541., 1178.,  182.,
     #               36., 17400., 16750., 4021., 2728., 1271.,  219.,
     #               37., 18410., 17740., 4265., 2924., 1369.,  262.,
     #               38., 19450., 18760., 4516., 3127., 1469.,  307.,
     #               39., 20520., 19800., 4773., 3334., 1570.,  353.,
     #               40., 21610., 20880., 5037., 3547., 1669.,  402.,
     #               41., 22740., 21980., 5307., 3765., 1766.,  451.,
     #               42., 23890., 23120., 5585., 3990., 1869.,  505.,
     #               43., 25080., 24290., 5874., 4225., 1978.,  565.,
     #               44., 26300., 25490., 6165., 4460., 2084.,  622.,
     #               45., 27550., 26720., 6466., 4705., 2197.,  684.,
     #               46., 28830., 27970., 6770., 4951., 2307.,  744.,
     #               47., 30140., 29270., 7092., 5213., 2431.,  817.,
     #               48., 31490., 30600., 7421., 5481., 2558.,  891.,
     #               49., 32870., 31960., 7758., 5755., 2688.,  969.,
     #               50., 34290., 33350., 8104., 6036., 2821., 1048.,
     #               51., 35730., 34780., 8457., 6323., 2957., 1131.,
     #               52., 37210., 36230., 8819., 6617., 3096., 1216.,
     #               53., 38730., 37730., 9190., 6917., 3239., 1304.,
     #               54., 40270., 39250., 9569., 7224., 3386., 1394.,
     #               55., 41860., 40820., 9958., 7540., 3537., 1489.,
     #               56., 43480., 42410.,10360., 7862., 3692., 1586.,
     #               57., 45130., 44040.,10760., 8189., 3850., 1685.,
     #               58., 46810., 45710.,11180., 8521., 4007., 1783.,
     #               59., 48530., 47400.,11590., 8855., 4162., 1878.,
     #               60., 50290., 49140.,12020., 9199., 4326., 1980.,
     #               61., 52080., 50910.,12470., 9551., 4493., 2085.,
     #               62., 53910., 52720.,12920., 9908., 4664., 2193.,
     #               63., 55780., 54560.,13370.,10270., 4838., 2303.,
     #               64., 57700., 56450.,13850.,10650., 5025., 2424.,
     #               65., 59640., 58380.,14330.,11030., 5207., 2541.,
     #               66., 61620., 60330.,14810.,11400., 5383., 2649.,
     #               67., 63640., 62330.,15310.,11790., 5571., 2769.,
     #               68., 65700., 64370.,15820.,12190., 5764., 2892.,
     #               69., 67810., 66450.,16340.,12590., 5961., 3018.,
     #               70., 69950., 68570.,16870.,13000., 6161., 3146.,
     #               71., 72150., 70740.,17430.,13440., 6380., 3291.,
     #               72., 74390., 72960.,17990.,13870., 6602., 3437.,
     #               73., 76670., 75210.,18570.,14310., 6827., 3584.,
     #               74., 78990., 77510.,19150.,14760., 7055., 3734.,
     #               75., 81360., 79850.,19750.,15210., 7288., 3886.,
     #               76., 83780., 82240.,20360.,15670., 7525., 4040.,
     #               77., 86230., 84670.,20980.,16140., 7766., 4197.,
     #               78., 88740., 87150.,21610.,16610., 8009., 4354.,
     #               79., 91290., 89680.,22260.,17090., 8259., 4516.,
     #               80., 93890., 92250.,22920.,17580., 8517., 4683.,
     #               81., 96540., 94880.,23600.,18080., 8780., 4854.,
     #               82., 99250., 97550.,24290.,18580., 9048., 5026.,
     #               83., 102000.,100300.,24990.,19100.,9321., 5202.,
     #               84., 104800.,103100.,25710.,19610.,9599., 5380.,
     #               85., 107700.,105900.,26450.,20140.,9882., 5560.,
     #               86., 110600.,108800.,27190.,20670.,10170.,5744.,
     #               87., 113500.,111700.,27960.,21210.,10470.,5931.,
     #               88., 116600.,114700.,28740.,21760.,10770.,6121.,
     #               89., 119700.,117800.,29540.,22310.,11070.,6313.,
     #               90., 122800.,120900.,30350.,22880.,11380.,6508.,
     #               91., 126000.,124000.,31180.,23440.,11700.,6701.,
     #               92., 129300.,127300.,32030.,24020.,12020.,6899.,
     #               93., 132600.,130600.,32890.,24600.,12350.,7100.,
     #               94., 136000.,133900.,33780.,25180.,12680.,7300.,
     #               95., 139400.,137300.,34670.,25780.,13020.,7509.,
     #               96., 143000.,140800.,35590.,26390.,13370.,7718.  /
C **********************************************************************
C     target data    Z     NT   DENSITY
      DATA TARDAT /  1., 0.005,  0.089,
     #               2., 0.003, 0.1785,
     #               3., 4.634, 534.,
     #               4., 12.37, 1850.,
     #               5., 13.04, 2340.,
     #               6., 11.3,  2250.,
     #               7., .005,  1.25,
     #               8., .005,  1.43,
     #               9., .005,  1.70,
     #              10., .003,  0.899,
     #              11., 2.543, 971.,
     #              12., 4.307, 1738.,
     #              13., 6.03,  2702.,
     #              14., 4.997, 2330.,
     #              15., 3.539, 1820.,
     #              16., 3.888, 2070.,
     #              17., 0.005, 3.214,
     #              18., 0.003, 1.784,
     #              19., 1.328, 862.,
     #              20., 2.329, 1550.,
     #              21., 4.005, 2989.,
     #              22., 5.66,  4500.,
     #              23., 7.224, 6110.,
     #              24., 8.329, 7190.,
     #              25., 8.003, 7300.,
     #              26., 8.492, 7874.,
     #              27., 9.096, 8900.,
     #              28., 9.136, 8902.,
     #              29., 8.46,  8940.,
     #              30., 6.570, 7133.,
     #              31., 5.100, 5904.,
     #              32., 4.44,  5350.,
     #              33., 4.606, 5730.,
     #              34., 3.654, 4790.,
     #              35., 2.352, 3120.,
     #              36., 0.003, 3.733,
     #              37., 1.080, 1532.,
     #              38., 1.746, 2540.,
     #              39., 3.028, 4469.,
     #              40., 4.23,  6400.,
     #              41., 5.556, 8570.,
     #              42., 6.416, 10220.,
     #              43., 7.141, 11500.,
     #              44., 7.395, 12410.,
     #              45., 7.264, 12410.,
     #              46., 6.803, 12020.,
     #              47., 5.86,  10500.,
     #              48., 4.635, 8650.,
     #              49., 3.835, 7310.,
     #              50., 2.917, 5750.,
     #              51., 3.310, 6691.,
     #              52., 2.945, 6240.,
     #              53., 2.340, 4930.,
     #              54., 0.003, 5.89,
     #              55., 0.849, 1873.,
     #              56., 1.535, 3500.,
     #              57., 2.665, 6145.,
     #              58., 2.861, 6657.,
     #              59., 2.895, 6773.,
     #              60., 2.839, 6800.,
     #              61., 2.999, 7220.,
     #              62., 3.012, 7520.,
     #              63., 2.078, 5243.,
     #              64., 3.026, 7900.,
     #              65., 3.16,  8330.,
     #              66., 3.169, 8550.,
     #              67., 3.212, 8795.,
     #              68., 3.265, 9066.,
     #              69., 3.323, 9321.,
     #              70., 2.424, 6965.,
     #              71., 3.387, 9840.,
     #              72., 4.491, 13310.,
     #              73., 5.53,  16600.,
     #              74., 6.323, 19300.,
     #              75., 6.799, 21020.,
     #              76., 7.147, 22570.,
     #              77., 7.025, 22420.,
     #              78., 6.623, 21450.,
     #              79., 5.90,  19300.,
     #              80., 4.067, 13546.,
     #              81., 3.492, 11850.,
     #              82., 3.30,  11340.,
     #              83., 2.809, 9747.,
     #              84., 2.686, 9320.,
     #              85., 0.005, 9.730,
     #              86., 0.003, 9.730,
     #              87., 0.005, 9.730,
     #              88., 1.333, 5000.,
     #              89., 2.672, 10070.,
     #              90., 2.94,  11200.,
     #              91., 4.01,  15370.,
     #              92., 4.73,  18700.,
     #              93., 5.15,  20250.,
     #              94., 4.90,  19840.,
     #              95., 3.39,  13670.,
     #              96., 3.29,  13510.,
     #              6.6, 6.08,  1385./
C **********************************************************************
C MFA
c      CALL DATIME(IY, ID)
C MFE
C PCA
C     CALL GETDAT(IYEAR,IMONTH,IDAY)
C     CALL GETTIM(IHOUR,IMINUTE,ISECOND,IHUND)
C     ISL(1) = IYEAR
C     ISL(2) = IMONTH
C     ISL(3) = IDAY
C     ISL(4) = IHOUR
C     ISL(5) = IMINUTE
C     ISL(6) = ISECOND
C PCE
      PI=3.1415
      U0=13.6
      MC2=511006.
      S0=7.038E+08
      ALPH=7.2993E-3
C **********************************************************************
C *** INPUT*
C **********************************************************************
      CALL INPUT
C **********************************************************************
1     CONTINUE
      CALL MENUE
C **********************************************************************
C
C *** IF LOOP...
      IF( I_LOOP .GT. 0 ) THEN
         WRITE(6,*) ' '
         WRITE(6,*) 'Loop parameters:'
      ENDIF
C
      IF( I_LOOP .EQ. 1 ) THEN
        ZFMIN = 30.
        DELZF = 10.
        CALL PRSO('Enter minimum projectile Z', ZFMIN, 2)
        IF( ZFMIN .LT. 1. ) THEN
          ZFMIN = 1.
          WRITE(6,*) '<I>: Min. projectile Z set to ZF(min) = 1! '
        ENDIF
        IF( ZFMIN .GT. 96. ) THEN
          ZFMIN = 96.
          WRITE(6,*) '<I>: Min. projectile Z set to ZF(max) = 96! '
        ENDIF
        IF( ZFMIN .LT. 29. ) 
     #    WRITE(6,*) '<W>: GLOBAL is designed for ZF > 28!'
        IF( NINT(ZFMIN) .LE. QIN ) THEN
          WRITE(6,*) '<W>: ZF lower than or equal to Q!'
          CALL PILO('Enter new Q state', QIN )
        ENDIF 
        IF( NINT(ZFMIN) .LE. QIN ) THEN
          QIN = 0
          WRITE(6,*) '<I>: Q set to zero!'
        ENDIF
        CALL PRSO('Enter projectile-Z steps  ', DELZF, 2)
        IF( DELZF .LT. 1. ) THEN 
          DELZF = 5.
          WRITE(6,*) '<I>: Z steps set to 5!'
        ENDIF
      ELSEIF( I_LOOP .EQ. 2 ) THEN
        EMIN =  100.
        DELE =  100.
        CALL PRSO('Enter minimum projectile energy (MeV/u)', EMIN, 4)
        IF( EMIN .LT. 30. ) THEN
          EMIN = 30.
          WRITE(6,*) '<I>: E(min) set to 30 MeV/u!'
        ENDIF
        IF( EMIN .GT. 2000. ) THEN
          EMIN = 2000.
          WRITE(6,*) '<I>: E(min) set to 2000 MeV/u!'
        ENDIF
        CALL PRSO('Enter projectile-energy steps (MeV/u) ', DELE, 4)
        IF( DELE .LT. 1. ) THEN 
          DELE = 10.
          WRITE(6,*) '<I>: Energy steps set to 10 MeV/u!'
        ENDIF
      ELSEIF( I_LOOP .EQ. 3 ) THEN
        QMIN = 0.
        DELQ = 5.
        CALL PILO('Enter minimum incident-electron number', NINT(QMIN))
        IF( QMIN .LT. 0. ) THEN
          QMIN = 0.
          WRITE(6,*) '<I>: Q(min) set to zero!'
        ENDIF
        IF( QMIN .GT. 28. ) THEN 
          QMIN = 28.
          WRITE(6,*) '<I>: Q(min) set to 28!'
        ENDIF
        IF( QMIN .GE. ZF ) THEN 
          QMIN = 0.
          WRITE(6,*) '<W>: Q larger than or equal to Zp ! '
          WRITE(6,*) '<I>: Q(min) set to zero!'
        ENDIF 
        CALL PILO('Enter Q-state steps                   ',NINT(DELQ))
        IF( DELQ .LT. 1. ) THEN 
          DELQ = 2.
          WRITE(6,*) '<I>: Q steps set to 2!'
        ENDIF
      ELSEIF( I_LOOP .EQ. 4 ) THEN
        ZTMIN = 5.
        DELZT = 10.
        CALL PRSO('Enter minimum target Z', ZTMIN, 2)
        IF( ZTMIN .LT. 1. ) THEN 
          ZTMIN = 4.
          WRITE(6,*) '<I>: Z(min) set to 4!'
        ENDIF
        IF( ZTMIN .GT. 96. ) THEN 
          ZTMIN = 96.
          WRITE(6,*) '<I>: Z(min) set to 96!'
        ENDIF
        CALL PRSO('Enter target-Z steps  ', DELZT, 2)
        IF( DELZT .LT. 1. ) THEN 
          DELZT = 5.
          WRITE(6,*) '<I>: Charge steps set to 5!'
        ENDIF
      ELSEIF( I_LOOP .EQ. 5 ) THEN
        DTMIN = 100.
        DELDT = 100.
        CALL PRSO('Enter minimum target thickness (mg/cm^2)', DTMIN, 4)
        IF( DTMIN .LE. 0. ) THEN 
          DTMIN = 10.
          WRITE(6,*) '<I>: Min.ickness set to 10 mg/cm^2!'
        ENDIF
        CALL PRSO('Enter target-thickness steps (mg/cm^2)  ', DELDT, 4)
        IF( DELDT .LE. 0. ) THEN 
          DELDT = 10.
          WRITE(6,*) '<I>: Thickness steps set to 10 mg/cm^2!'
        ENDIF
      ENDIF
C
      DTARGET0 = DTARGET
      IF( I_CHAR .EQ. 1 ) DTARGET = 10000.
      ZF0 = ZF
      AF0 = AF
      QIN0 = QIN
      ZT0 = ZT
      AT0 = AT
      LOOP = 0
      ENMAX = EN0
C
5     CONTINUE
      EN = EN0
      I_EN = 0
C
C *** PROJECTILE
C
      IF( I_LOOP .EQ. 1 ) THEN
        ZF = ZFMIN + FLOAT(LOOP) * DELZF
        CALL ELEMENT( ZF, AF, CPRO, 3, IRC )
      ENDIF
C
      IP = INT(ZF)
C
      IF( I_LOOP .EQ. 2 ) THEN
        EN = EMIN + FLOAT(LOOP) * DELE
        EN0 = EN
      ENDIF
C
      J = MAX(10, MIN(28, MAX(JQST(0), JQST(1), JQST(2), JQST(3),
     #            JQST(4), JQST(5), JQST(6), JQST(7), JQST(8), JQST(9),
     #            QIN)+8) )
C 
      IF( I_LOOP .EQ. 3 ) THEN
        QIN = INT( QMIN + FLOAT(LOOP) * DELQ)
        J = 28
      ENDIF
C
C *** TARGET
C
      IF( I_LOOP .EQ. 4 ) THEN
        ZT = ZTMIN + FLOAT(LOOP) * DELZT
        CALL ELEMENT( ZT, AT, CTAR, 1, IRC )
      ENDIF
C
      IT = INT(ZT)
      IF( ZT .EQ. 6.6 ) IT = 97
C
      IF( I_LOOP .EQ. 5 ) THEN
        DTARGET = DTMIN + FLOAT(LOOP) * DELDT
      ENDIF
C
      IF( LOOP .EQ. 0 ) THEN
       WRITE(6,*)
     #   '--------------------------------------------------------',
     #   '-----------------------'
       WRITE(6,*)
     #   'GLOBAL: Q-states of heavy ions behind matter layers',
     #   '               Version ', CVERS
       WRITE(6,*)
     #   '--------------------------------------------------------',
     #   '-----------------------'
      ENDIF
C
C MFA
      IF( ( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) .and. LOOP .EQ. 0  ) THEN
        WRITE(8,11) CVERS, ISL(3),ISL(2),ISL(1),ISL(4),ISL(5),ISL(6)
 11    FORMAT(' ***** Global *****  Version ', A5, ' **** ',
     #  I2,'.',I2,'.',I4,' **** ', I2,':',I2,':',I2,' h' )
      ENDIF
C MFE
C
C PCA-E
C
      ZF   = PRODAT(1,IP)
      U1S  = PRODAT(2,IP)
      U1SS = PRODAT(3,IP)
      U2S  = PRODAT(4,IP)
      U2P  = PRODAT(5,IP)
      U3S  = PRODAT(6,IP)
      U3D  = PRODAT(7,IP)
      ZT   = TARDAT(1,IT)
      NT   = TARDAT(2,IT)
      DE   = TARDAT(3,IT)
      IF( NT .LT. 0.01 ) THEN
        I_GAS = 1
      ELSE
        I_GAS = 0
      ENDIF
C
C *** Final charge fraction calc. and printout
C
      DO 30 I = 0, IMAX-1
        CALL C_INCH( JQST(I), CI )
        C = CI
        IF( JQST(I) .GE. 0 ) THEN
C MFA
          CQST(I) = ' Q('//C//') '
C MFE
C PCA
C         CQST(I) = ' Q'//C//' '
C PCE
        ELSE
          CQST(I) = ' '
        ENDIF
30    CONTINUE
C
      IF( LOOP .EQ. 0 ) THEN
       IF( I_CHAR .EQ. 1 ) THEN
         DTARG = 0.
       ELSE
         DTARG = DTARGET
       ENDIF
       IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         WRITE(6,*)
         IF( I_LOOP .EQ. 0 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(6,36) INT(ZF), AF, QIN, EN0, INT(ZT), AT
           ELSE
              WRITE(6,31) INT(ZF), AF, QIN, EN0, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 1 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(6,322) QIN, EN0, INT(ZT), AT
           ELSE
              WRITE(6,32) QIN, EN0, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 2 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(6,333) INT(ZF), AF, QIN, INT(ZT), AT
           ELSE
              WRITE(6,33) INT(ZF), AF, QIN, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 3 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(6,344) INT(ZF), AF, EN0, INT(ZT), AT
           ELSE
              WRITE(6,34) INT(ZF), AF, EN0, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 4 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(6,355) INT(ZF), AF, QIN, EN0
           ELSE
              WRITE(6,35) INT(ZF), AF, QIN, EN0, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 5 ) THEN
           WRITE(6,36) INT(ZF), AF, QIN, EN0, INT(ZT), AT
         ENDIF
       ENDIF
       IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
C MFA
           WRITE(8,*)
C MFE
C PCA-E
         IF( I_LOOP .EQ. 0 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(8,36) INT(ZF), AF, QIN, EN0, INT(ZT), AT
           ELSE
C MFA
              WRITE(8,31) INT(ZF),AF, QIN, EN0, INT(ZT), AT, DTARG
C MFE
C PCA-E
           ENDIF
         ELSE IF( I_LOOP .EQ. 1 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(8,322) QIN, EN0, INT(ZT), AT
           ELSE
              WRITE(8,32) QIN, EN0, INT(ZT), AT, DTARG
           ENDIF
        ELSE IF( I_LOOP .EQ. 2 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(8,333) INT(ZF), AF, QIN, INT(ZT), AT
           ELSE
              WRITE(8,33) INT(ZF), AF, QIN, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 3 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(8,344) INT(ZF), AF, EN0, INT(ZT), AT
           ELSE
              WRITE(8,34) INT(ZF), AF, EN0, INT(ZT), AT, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 4 ) THEN
           IF( I_CHAR .EQ. 1 ) THEN
              WRITE(8,355) INT(ZF), AF, QIN, EN0
           ELSE
              WRITE(8,35) INT(ZF), AF, QIN, EN0, DTARG
           ENDIF
         ELSE IF( I_LOOP .EQ. 5 ) THEN
           WRITE(8,36) INT(ZF), AF,QIN, EN0, INT(ZT), AT
         ENDIF
       ENDIF
       IF( I_OUTP .EQ. 3 ) THEN
         WRITE(6,*)
         WRITE(6,31) INT(ZF),AF, QIN, EN0,INT(ZT),AT, DTARG
       ENDIF
      ENDIF
C
31    FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,') at E =',F6.1,' MeV/u'
     #      ,' on (Z=' ,I2,', A=', F5.1,', D=',G8.2,' mg/cm^2)')
32    FORMAT(' (Projectile: Qe=',I2,')  at E =',F6.1,' MeV/u'
     #      ,' on (Z=' ,I2,', A=', F5.1,', D=',G8.2,' mg/cm^2)')
322   FORMAT(' (Projectile: Qe=',I2,')  at E =',F6.1,' MeV/u'
     #      ,' on (Z=' ,I2,', A=', F5.1,')')
33    FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,') on (Z=' ,I2,
     #       ', A=', F5.1,', D=',G8.2,' mg/cm^2)')
333   FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,') on (Z=' ,I2,
     #       ', A=', F5.1,')')
34    FORMAT(' (Z=',I2, ', A=',F4.0, ') at E =',F6.1,' MeV/u'
     #      ,' on (Z=' ,I2,', A=', F5.1,', D=',G8.2,' mg/cm^2)')
344   FORMAT(' (Z=',I2, ', A=',F4.0, ') at E =',F6.1,' MeV/u'
     #      ,' on (Z=' ,I2,', A=', F5.1,')')
35    FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,')  at E =',F6.1,
     #      ' MeV/u', ' on (target:  D=',G8.2,' mg/cm^2)')
355   FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,')  at E =',F6.1,
     #      ' MeV/u')
36    FORMAT(' (Z=',I2, ', A=',F4.0, ', Qe=',I2,')  at E =',F6.1,
     #      ' MeV/u', ' on (Z=' ,I2,', A=', F5.1,')')
*
      WRITE(6,*) ' '
      IF( LOOP .EQ. 0 ) THEN
       IF( I_CHAR .EQ. 0 .OR. I_CHAR .EQ. 1 .OR.
     #   ( I_CHAR .EQ. 2 .AND. I_LOOP .EQ. 0 ) ) THEN
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
          IF( I_CHAR .EQ. 0 ) WRITE(6,*) 'Q-states at target exit:'
          IF( I_CHAR .EQ. 1 ) WRITE(6,*) 'Equilibrium Q-states:'
          IF( I_CHAR .EQ. 2 ) WRITE(6,*) 'Q-states evolution:'
          WRITE(6,*)
          IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 ) THEN
            WRITE(6,101) (CQST(I), I = 0, MIN(4,IMAX-1))
          ELSEIF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 2 ) THEN
            WRITE(6,102) (CQST(I), I = 0, MIN(4,IMAX-1))
          ELSEIF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,103) (CQST(I), I = 0, MIN(4,IMAX-1))
          ENDIF
          IF( I_LOOP .EQ. 1 ) WRITE(6,104) (CQST(I),
     #      I = 0, MIN(4,IMAX-1))
          IF( I_LOOP .EQ. 2 ) WRITE(6,105) (CQST(I),
     #      I = 0, MIN(4,IMAX-1))
          IF( I_LOOP .EQ. 3 ) WRITE(6,106) (CQST(I),
     #      I = 0, MIN(4,IMAX-1))
          IF( I_LOOP .EQ. 4 ) WRITE(6,107) (CQST(I),
     #      I = 0, MIN(4,IMAX-1))
          IF( I_LOOP .EQ. 5 ) WRITE(6,108) (CQST(I),
     #      I = 0, MIN(4,IMAX-1))
          IF( IMAX .GE. 5 ) THEN
            IF( I_CHAR .EQ. 2 ) THEN 
             WRITE(6,110) (CQST(I), I = 5, IMAX-1)
            ELSE
             WRITE(6,109) (CQST(I), I = 5, IMAX-1)
            ENDIF
          ENDIF
          WRITE(6,*)
        ENDIF
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
C MFA
          WRITE(8,*)
C MFE
C PCA-E
          IF( I_CHAR .EQ. 0 ) WRITE(8,*) 'Q-states at target exit:'
          IF( I_CHAR .EQ. 1 ) WRITE(8,*) 'Equilibrium Q-states:'
C MFA
          IF( I_CHAR .EQ. 2 ) WRITE(8,*) 'Q-states evolution:'
          WRITE(8,*)
C MFE
C PCA-E
          IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 ) THEN
            WRITE(8,111) (CQST(I), I = 0, IMAX-1)
          ELSEIF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 2 ) THEN
            WRITE(8,112) (CQST(I), I = 0, IMAX-1 )
          ELSEIF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,113) (CQST(I), I = 0, IMAX-1 )
          ENDIF
          IF( I_LOOP .EQ. 1 ) WRITE(8,114) (CQST(I),
     #      I = 0, IMAX-1 )
          IF( I_LOOP .EQ. 2 ) WRITE(8,115) (CQST(I),
     #      I = 0, IMAX-1 )
          IF( I_LOOP .EQ. 3 ) WRITE(8,116) (CQST(I),
     #      I = 0, IMAX-1 )
          IF( I_LOOP .EQ. 4 ) WRITE(8,117) (CQST(I),
     #      I = 0, IMAX-1 )
          IF( I_LOOP .EQ. 5 ) WRITE(8,118) (CQST(I),
     #      I = 0, IMAX-1 )
        ENDIF
       ENDIF
      ENDIF
101   FORMAT(' D_eq(mg/cm^2)        Eout     ',A7, 3X, A7, 3X, A7, 
     #                                3X, A7, 3X, A7)
102   FORMAT('  D(mg/cm^2)   ',A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
103   FORMAT(' D(mg/cm^2)   D_eq    Eout     ',A7, 3X, A7, 3X, A7, 
     #                                3X, A7, 3X, A7)
104   FORMAT('   Z_p        D_eq    Eout     ',A7, 3X, A7, 3X, A7,
     #                                3X, A7, 3X, A7)
105   FORMAT('  E(MeV/u)    D_eq    Eout     ',A7, 3X, A7, 3X, A7, 
     #                                3X, A7, 3X, A7)
106   FORMAT('    Q         D_eq    Eout     ',A7, 3X, A7, 3X, A7,
     #                                3X, A7, 3X, A7)
107   FORMAT('   Z_t        D_eq    Eout     ',A7, 3X, A7, 3X, A7, 
     #                                3X, A7, 3X, A7)
108   FORMAT(' D(mg/cm^2)   D_eq    Eout     ',A7, 3X, A7, 3X, A7,
     #                                3X, A7, 3X, A7)
109   FORMAT('                               ',A7, 3X, A7, 3X, A7, 
     #                                3X, A7, 3X, A7)
110   FORMAT('               ',A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
C
111   FORMAT('   D_eq     Eout      ',  A7, 3X, A7, 3X, A7, 3X, A7, 
     #             3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
112   FORMAT('    D          ',A7, 3X, A7, 3X, A7, 3X, A7, 
     #             3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
113   FORMAT('    D        D_eq    Eout      ',A7, 3X, A7, 3X, A7,  
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
114   FORMAT('   Z_p    D_eq    Eout         ',A7, 3X, A7, 3X, A7,  
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
115   FORMAT('     E       D_eq    Eout      ',A7, 3X, A7, 3X, A7, 
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
116   FORMAT('    Q     D_eq    Eout         ',A7, 3X, A7, 3X, A7,  
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
117   FORMAT('   Z_t    D_eq    Eout         ',A7, 3X, A7, 3X, A7,  
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
118   FORMAT('    D        D_eq    Eout      ',A7, 3X, A7, 3X, A7, 
     #     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
C
      IF( I_CHAR .EQ. 2 .AND. I_LOOP .NE. 0 ) THEN
       IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( LOOP .EQ. 0 ) WRITE(6,*) 'Q-state evolution:'
         IF( LOOP .EQ. 0 ) WRITE(6,*)
         IF( I_LOOP .EQ. 1 ) WRITE(6,121) NINT(ZF), NINT(AF)
         IF( I_LOOP .EQ. 2 ) WRITE(6,122) EN
         IF( I_LOOP .EQ. 3 ) WRITE(6,123) QIN
         IF( I_LOOP .EQ. 4 ) WRITE(6,124) NINT(ZT), NINT(AT)
         IF( I_LOOP .EQ. 5 ) WRITE(6,125) DTARGET
         WRITE(6,119) (CQST(I), I = 0, MIN(4,IMAX-1))
         WRITE(6,120) (CQST(I), I = 5, IMAX-1)
119      FORMAT(' D(mg/cm^2)    ',A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
120      FORMAT('               ',A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
       ENDIF
       IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
C MFA
         WRITE(8,*)
         WRITE(8,*)
         WRITE(8,*) 'Q-state evolution:'
C MFE
C PCA-E
         IF( I_LOOP .EQ. 1 ) WRITE(8,121) NINT(ZF), NINT(AF)
         IF( I_LOOP .EQ. 2 ) WRITE(8,122) EN
         IF( I_LOOP .EQ. 3 ) WRITE(8,123) QIN
         IF( I_LOOP .EQ. 4 ) WRITE(8,124) NINT(ZT), NINT(AT)
         IF( I_LOOP .EQ. 5 ) WRITE(8,125) DTARGET
C MFA
         WRITE(8,*)
C MFE
C PCA-E
         WRITE(8,126) (CQST(I),  I = 0, IMAX-1 )
       ENDIF
121    FORMAT(' Projectile: Z = ', I2, ', A = ', I3 )
122    FORMAT(' Energy:  E = ', F6.1, ' MeV/u' )
123    FORMAT(' Q-state: Q = ', I2, '+')
124    FORMAT(' Target: Z = ', I2, ', A = ', I3 )
125    FORMAT(' Thickness: D = ', G9.4, ' mg/cm^2')
126    FORMAT(' D(mg/cm^2)   ',A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7,
     #                     3X, A7, 3X, A7, 3X, A7, 3X, A7, 3X, A7)
      ENDIF
C
C *** Determine equilibrium thickness and step size
C
      ENKEEP = EN
      CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
      IF( IRC .NE. 0 ) GOTO 9000
C
      D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
      IF( J .LE. 10 ) THEN
        AUX = DE / NT / LIS
      ELSE
        AUX = DE / NT / MIS
      ENDIF
      IF (AUX .LT. 0.03) THEN
        AUX = 0.01
      ELSE IF (AUX .GE. 0.03 .AND. AUX .LT. 0.08) THEN
        AUX = 0.05
      ELSE IF (AUX .GE. 0.08 .AND. AUX .LT. 0.3) THEN
        AUX = 0.1
      ELSE IF (AUX .GE. 0.3 .AND. AUX .LT. 0.8) THEN
        AUX = 0.5
      ELSE
        AUX = 1.
      ENDIF
      DELT = .01 * AUX
      IF ( NT .LT. 0.01 ) DELT = 0.002 * AUX
C
C *** RESET
140   DO 150 I=0, 30
        X(I)=0.
        DX(I)=0.
150   CONTINUE
      X(QIN)=1.
      DO 160, I=0, 28
        SA(I)=0.
        FA(I)=0.
160   CONTINUE
      I_WRITE_FAC = 1
C
C *** start of charge-fraction integration
C     RESTART FOR NEGATIVE CHARGE STATES
      ENKEEP = 0.
      TX=NT*DELT/100./DE
      TX0=TX
      TX1=5.*TX
      TX2=10.*TX
      TX3=100.*TX
      TX4=1000.*TX
      TX5=2000.*TX
C
C *** first integration step
C
      TX=TX0
C
      DO 200, I = 0, J
        SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
        SA(I)=TX*SA(I)
        FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
200   CONTINUE
C
      DO 205, IR=1, 400
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
C
        IF( EN .LE. 30. ) THEN
          I_EN = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
C
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF .OR. ENKEEP .LT. EN ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 210, I = 0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
210       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0) = TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1) = TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2) = TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)+A(17)
     #          *X(4))*FA(2)
        DO 215, I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
215     CONTINUE
C
        DO 220, I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I) = 0.
220     CONTINUE
        DO 225, I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
225     CONTINUE
        DO 230, I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
230     CONTINUE
C
C  **   CHECK CHARGE STATES
        DO 240 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241) DELT
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
240     CONTINUE
241     FORMAT( ' <I>: Target step size ',G7.1, ' too large! ',
     #                ' Restart of calculations!' )
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
            I_EQUI = 1
        ELSE
            I_EQUI = 0
        ENDIF
C
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF, D_EQ, EN,   
     #                          (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0, D_EQ, EN, 
     #                          (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN, D_EQ, EN, 
     #                          (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT, D_EQ, EN,
     #                          (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,D_EQ,EN,
     #                          (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2  ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC ) .EQ. 0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
252     FORMAT(2X,F4.0, 5X, F7.1, 3X, F6.1, 2X, F8.5, 2X, F8.5, 2X,
     #         F8.5, 2X, F8.5, 2X, F8.5)
253     FORMAT(2X,F6.0, 3X, F7.1, 3X, F6.1, 2X, F8.5, 2X, F8.5, 2X, 
     #         F8.5, 2X, F8.5, 2X, F8.5)
254     FORMAT(2X,I3,   6X, F7.1, 3X, F6.1, 2X, F8.5, 2X, F8.5, 2X, 
     #         F8.5, 2X, F8.5, 2X, F8.5)
255     FORMAT(3X,F4.0, 4X, F7.1, 3X, F6.1, 2X, F8.5, 2X, F8.5, 2X,
     #         F8.5, 2X, F8.5, 2X, F8.5)
256     FORMAT(F9.3,    2X, F7.1, 3X, F6.1, 2X, F8.5, 2X, F8.5, 2X, 
     #         F8.5, 2X, F8.5, 2X, F8.5)
257     FORMAT(29X,      F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5)
258     FORMAT(F9.3,    11X, F7.1, 2X, F8.5, 2X, F8.5, 2X, 
     #         F8.5, 2X, F8.5, 2X, F8.5)
261     FORMAT(F9.3, 3x, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5)
262     FORMAT(12X,      F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5)
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                                    (X(JQST(I)), I = 0,IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 )  THEN
            WRITE(8,276) DELT*FLOAT(IR), D_EQ, EN,
     #                                    (X(JQST(I)), I = 0,IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,      D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,      D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET, D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC ) .EQ. 0 
     #       .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
272     FORMAT(F6.0, 2X, F7.1, 2X, F6.0, 6X, F8.5, 2X, F8.5, 2X, F8.5,  
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
273     FORMAT(F8.0, 2X, F7.1, 2X, F6.0, 4X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
274     FORMAT(I5,   2X, F7.1, 2X, F6.0, 7X, F8.5, 2X, F8.5, 2X, F8.5,  
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
275     FORMAT(F6.1, 2X, F7.1, 2X, F6.0, 6X, F8.5, 2X, F8.5, 2X, F8.5,  
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
276     FORMAT(F9.3, 2X, F7.1, 2X, F6.0, 3X, F8.5, 2X, F8.5, 2X, F8.5,  
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
278     FORMAT(F9.3, 2X, F6.0, 3X, F8.5, 2X, F8.5, 2X, F8.5,  
     #   2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 
     #   2X, F8.5)
281     FORMAT(F9.3, 3x, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5,
     #     2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5, 2X, F8.5)
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
C
205   CONTINUE
C
C *** Change integration step
C
      TX=TX1
      DO 300, I=0, J
        SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
        SA(I)=TX*SA(I)
        FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
300   CONTINUE
C
      DO 305 IR=405, 4000, 5
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
        IF( EN .LE. 30. ) THEN
          I_EN = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 310, I=0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
310       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0)=TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1)=TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2)=TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)
     #       +A(17)*X(4))*FA(2)
        DO 315 I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
315     CONTINUE
        DO 320 I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I)=0.
320     CONTINUE
        DO 325 I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
325     CONTINUE
        DO 330 I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
330     CONTINUE
C
C  **   CHECK CHARGE STATES
        DO 340 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241) DELT
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
340     CONTINUE
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
             I_EQUI = 1
        ELSE
             I_EQUI = 0
        ENDIF
C
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN,  (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,   D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*5 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                   (X(JQST(I)), I = 0, IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,276) DELT*FLOAT(IR),  D_EQ, EN, 
     #                   (X(JQST(I)), I = 0,IMAX-1 )
C           WRITE(8,271)     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET,D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*5 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
C
305   CONTINUE
C
C *** Change integration step
C
      TX=TX2
      DO 400 I=0, J
        SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
        SA(I)=TX*SA(I)
        FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
400   CONTINUE
C
      DO 405 IR=4010, 40000, 10
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
        IF( EN .LE. 30. ) THEN
          I_EN = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 410 I=0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
410       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0)=TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1)=TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2)=TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)
     #       +A(17)*X(4))*FA(2)
        DO 415 I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
415     CONTINUE
        DO 420 I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I)=0.
420     CONTINUE
        DO 425 I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
425     CONTINUE
        DO 430 I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
430     CONTINUE
C
C  **   CHECK CHARGE STATES
         DO 440 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241)
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
440     CONTINUE
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
            I_EQUI = 1
        ELSE
            I_EQUI = 0
        ENDIF
C
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN,  (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,   D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*10 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                   (X(JQST(I)), I = 0, IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,276) DELT*FLOAT(IR),  D_EQ, EN, 
     #                   (X(JQST(I)), I = 0,IMAX-1 )
C           WRITE(8,271)     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET,D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*10 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
C
405   CONTINUE
C
C *** Change integration step
C
      TX=TX3
      DO 500 I=0, J
        SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
        SA(I)=TX*SA(I)
        FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
500   CONTINUE
C
      DO 505 IR=40100, 400000, 100
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
        IF( EN .LE. 30. ) THEN
          I_EN = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 510 I=0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
510       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0)=TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1)=TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2)=TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)
     #       +A(17)*X(4))*FA(2)
        DO 515 I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
515     CONTINUE
        DO 520 I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I)=0.
520     CONTINUE
        DO 525 I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
525     CONTINUE
        DO 530 I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
530     CONTINUE
C
C  **   CHECK CHARGE STATES
        DO 540 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241) DELT
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
540     CONTINUE
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
            I_EQUI = 1
        ELSE
            I_EQUI = 0
        ENDIF
 
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN,  (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,   D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*100 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                   (X(JQST(I)), I = 0, IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,276) DELT*FLOAT(IR),  D_EQ, EN, 
     #                   (X(JQST(I)), I = 0,IMAX-1 )
C           WRITE(8,271)         D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET,D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*100 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
C
505   CONTINUE
C
C *** Change integration step
C
      TX=TX4
      DO 600 I=0, J
         SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
         SA(I)=TX*SA(I)
         FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
600   CONTINUE
C
      DO 605 IR=401000, 4000000, 1000
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
        IF( EN .LE. 30. ) THEN
          I_EN = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 610 I=0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
610       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0)=TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1)=TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2)=TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)
     #       +A(17)*X(4))*FA(2)
        DO 615 I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
615     CONTINUE
        DO 620 I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I)=0.
620     CONTINUE
        DO 625 I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
625     CONTINUE
        DO 630 I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
630     CONTINUE
C
C  **   CHECK CHARGE STATES
        DO 640 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241) DELT
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
640     CONTINUE
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
             I_EQUI = 1
        ELSE
             I_EQUI = 0
        ENDIF
C
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN,  (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,   D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*1000 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                   (X(JQST(I)), I = 0, IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,276) DELT*FLOAT(IR),  D_EQ, EN, 
     #                   (X(JQST(I)), I = 0,IMAX-1 )
C           WRITE(8,271)         D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET,D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*1000 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
605   CONTINUE
C
C *** Change integration step
C
      TX=TX5
C
      IMAXLOOP = NINT(MIN(2.E9, (DTARGET / DELT) * 1.1 ))
      DO 700 I=0, J
        SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
        SA(I)=TX*SA(I)
        FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
700   CONTINUE
C
      DO 705 IR = 4002000, IMAXLOOP, 2000
        IF( IR .EQ. IMAXLOOP )
     #    WRITE(6,*) '<I>: Max. step number 2*10**9 reached'
C       Adjust projectile energy, if necessary
        CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
        RG = RG - DELT*FLOAT(IR)
        CALL ENERGY( AF, ZF, AT, ZT, RG, EN )
        IF( EN .LE. 30. ) THEN
          I_EN  = 1
          WRITE(6,*) '<E>: Energy lower than 30 MeV/u! ',
     #               ' Calculations stopped!'
          GOTO 9000
        ENDIF
        EDIFF = 1.-LOG10(EN)**2 / 200.
        IF( EN .LT. ENKEEP * EDIFF ) THEN
          ENKEEP = EN
          CALL CROSS( U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
          IF( IRC .NE. 0 ) GOTO 9000
C
          DO 710 I=0, J
            SA(I)=A(6*I)+A(6*I+1)+A(6*I+2)+A(6*I+3)+A(6*I+4)+A(6*I+5)
            SA(I)=TX*SA(I)
            FA(I)=1.+SA(I)/2.+SA(I)*SA(I)/6.+SA(I)*SA(I)*SA(I)/24.
710       CONTINUE
          D_EQ = DE * 100. / NT * 4.6 / (KIB+ KCB/2.)
        ENDIF
C
        EA=0.
        SX=0.
        ZSQ=0.
        DX(0)=TX*(A(3)*X(0)+A(4)*X(1)+A(5)*X(2))*FA(0)
        DX(1)=TX*(A(8)*X(0)+A(9)*X(1)+A(10)*X(2)+A(11)*X(3))*FA(1)
        DX(2)=TX*(A(13)*X(0)+A(14)*X(1)+A(15)*X(2)+A(16)*X(3)
     #       +A(17)*X(4))*FA(2)
        DO 715 I=3, J
          DX(I)=TX*(A(6*I)*X(I-3)+A(6*I+1)*X(I-2)+A(6*I+2)*X(I-1)
     #         +A(6*I+3)*X(I)+A(6*I+4)*X(I+1)+A(6*I+5)*X(I+2))*FA(I)
715     CONTINUE
        DO 720 I=0, J
          IF( ABS(DX(I)) .LT. 9.999999E-21 ) DX(I)=0.
720     CONTINUE
        DO 725 I=0, J
          X(I)=X(I)+DX(I)
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          SX=SX+X(I)
          IF( ABS(SX) .LT. 1.E-20 ) SX = 0.
725     CONTINUE
        DO 730 I=0, J
          X(I)=X(I)/SX
          IF( ABS(X(I)) .LT. 1.E-20 ) X(I) = 0.
          EA=EA+I*X(I)
          ZSQ=ZSQ+(ZF-I)*(ZF-I)*X(I)
730     CONTINUE
C
C  **   CHECK CHARGE STATES
        DO 740 I=0, J
          IF( X(I) .LT. -1.E-5) THEN
            WRITE(6,*)
            WRITE(6,241) DELT
            WRITE(6,*)
            WRITE(8,*)
            WRITE(8,241) DELT
            WRITE(8,*)
            DELT = DELT / 10.
            I_WRITE_FAC = I_WRITE_FAC * 10
            GOTO 140
          ENDIF
740     CONTINUE
C
C  **   CHECK EQUILIBRIUM CHARGE STATES
        IF( DELT*FLOAT(IR) .GE. D_EQ ) THEN
            I_EQUI = 1
        ELSE
            I_EQUI = 0
        ENDIF
 
C  **   OUTPUT
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_CHAR .EQ. 1 .AND. 
     #         I_EQUI .EQ. 1 ) THEN
            WRITE(6,258) DELT*FLOAT(IR), EN, (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(6,256) DELT*FLOAT(IR), D_EQ, EN,  (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1) )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(6,252) ZF,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 2 )
     #      WRITE(6,253) EN0,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 3 )
     #      WRITE(6,254) QIN,       D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 4 )
     #      WRITE(6,255) ZT,        D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( I_LOOP .EQ. 5 )
     #      WRITE(6,256) DTARGET,   D_EQ, EN, 
     #                              (X(JQST(I)), I = 0, MIN(4,IMAX-1))
           IF( IMAX .GE. 6 ) WRITE(6,257) (X(JQST(I)), I = 5, IMAX-1)
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*2000 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(6,261) DELT*FLOAT(IR), (X(JQST(I)), I = 0,
     #                   MIN(4,IMAX-1))
            IF( IMAX .GE. 6 ) WRITE(6,262) (X(JQST(I)), I = 5, IMAX-1)
          ENDIF
         ENDIF
        ENDIF
C
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
         IF( (I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #       (I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) ) THEN
           IF( I_LOOP .EQ. 0 .AND. I_EQUI .EQ. 1 ) THEN
            WRITE(8,278) DELT*FLOAT(IR), EN, 
     #                   (X(JQST(I)), I = 0, IMAX-1 )
           ELSE IF( I_LOOP .EQ. 0 ) THEN
            WRITE(8,276) DELT*FLOAT(IR),  D_EQ, EN, 
     #                   (X(JQST(I)), I = 0,IMAX-1 )
C           WRITE(8,271)         D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           ENDIF
           IF( I_LOOP .EQ. 1 )
     #      WRITE(8,272) ZF,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 2 )
     #      WRITE(8,273) EN0,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 3 )
     #      WRITE(8,274) QIN,    D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 4 )
     #      WRITE(8,275) ZT,     D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
           IF( I_LOOP .EQ. 5 )
     #      WRITE(8,276) DTARGET,D_EQ, EN, (X(JQST(I)), I = 0, IMAX-1 )
C
         ELSE IF( I_CHAR .EQ. 2 ) THEN
          IF( MOD(IR, I_WRI*I_WRITE_FAC*2000 ).EQ.0 
     #        .OR. DELT*FLOAT(IR) .GE. DTARGET ) THEN
            WRITE(8,281) DELT*FLOAT(IR), (X(JQST(I)), I = 0, IMAX-1 )
          ENDIF
         ENDIF
        ENDIF
C
        IF(( I_CHAR .EQ. 0 .AND. DELT*FLOAT(IR) .GE. DTARGET ) .OR.
     #     ( I_CHAR .EQ. 1 .AND. I_EQUI .EQ. 1 ) .OR.
     #     ( I_CHAR .EQ. 2 .AND. DELT*FLOAT(IR) .GE. DTARGET ))GOTO 9000
705   CONTINUE
C
9000  CONTINUE
C
      IF( I_EQUI .EQ. 1 .AND. I_LOOP .GT. 0 .AND. I_CHAR .EQ. 1 ) THEN
C        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 )
C     #      WRITE(6,9021) FLOAT(IR) * DELT
C PCA COMMENT OUT THE TWO FOLLOWING LINES
C        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 )
C     #      WRITE(8,9021) FLOAT(IR) * DELT
C PCE
C9021    FORMAT('             <I>: Equilibrium target thickness:',
C     #       F7.1, ' mg/cm^2')
      ELSEIF( I_EQUI .LE. 0 .AND. I_LOOP .GT. 0 .AND. I_CHAR .EQ. 1 )
     #  THEN
        IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 ) THEN
            WRITE(6,*) '            <W>: Equilibrium target'//
     #       ' thickness not reached!'
        ENDIF
        IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 ) THEN
            WRITE(8,*) '            <W>: Equilibrium target'//
     #       ' thickness not reached!'
        ENDIF
      ENDIF
C
      CALL BBRANGE( AF, ZF, AT, ZT, EN0, RG )
      RG = RG - FLOAT(IR) * DELT
      CALL ENERGY( AF, ZF, AT, ZT, RG, EOUT )
C      IF( I_OUTP .EQ. 0 .OR. I_OUTP .EQ. 2 .AND. I_EN .NE. 1 )
C     #  WRITE(6,9023) EOUT
C PCA COMMENT OUT THE TWO FOLLOWING LINES
C      IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 .AND. I_EN .NE. 1 )
C     #  WRITE(8,9023) EOUT
C PCE
C9023   FORMAT('             <I>: Energy at target exit:', F7.1,
C     #   ' MeV/u')
C
      IF( I_LOOP .NE. 0 ) THEN
        LOOP = LOOP + 1
        IF( I_LOOP .EQ. 1 ) THEN
          IF( ZF+DELZF .GT. 96.) THEN
            ZF = ZF0
            AF = AF0
            GOTO 9100
          ENDIF
        ELSE IF( I_LOOP .EQ. 2 ) THEN
          IF( EN0+DELE .GT. ENMAX) THEN
            EN = ENMAX
            EN0 = ENMAX
            GOTO 9100
          ENDIF
        ELSE IF( I_LOOP .EQ. 3 ) THEN
          IF( QIN+DELQ .GE. INT(ZF/2.) .OR. QIN+DELQ .GT. 28. ) THEN
            QIN = QIN0
            GOTO 9100
          ENDIF
        ELSE IF( I_LOOP .EQ. 4 ) THEN
          IF( ZT+DELZT .GT. 96. ) THEN
            ZT = ZT0
            AT = AT0
            GOTO 9100
          ENDIF
        ELSE IF( I_LOOP .EQ. 5 ) THEN
          IF( DTARGET + DELDT .GT. DTARGET0 .OR. EN .LE. 30. )  THEN
            DTARGET = DTARGET0
            GOTO 9100
          ENDIF
        ENDIF
        GOTO 5
      ENDIF
 
C
9100  CONTINUE
      IF( I_CHAR .EQ. 1 ) DTARGET = DTARGET0
C
      CYES = 'N'
      WRITE(6,*)
      WRITE(6,*)
      CALL PYES('Write current setting to new INPUT file', CYES)
      IF( CYES .NE. 'N' .AND. CYES .NE. 'n' )  CALL OUTPUT
C
      CYES = 'Y'
      CALL PYES('ONCE MORE', CYES )
      IF( CYES .NE. 'N' .AND. CYES .NE. 'n' ) THEN
       WRITE(8,*)
       WRITE(8,*)
       LOOP = 0
       GOTO 1
      ENDIF
999   CONTINUE
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE CROSS(U1S, U1SS, U2S, U2P, U3S, U3D, EN, J, I_GAS, IRC)
C
      REAL*4 SC(3), SRKBEST, SRLU, SRMU, LMCKB, KCBNR, KCBRA
      REAL*4 LCBNR, LCBRA, MCBNR, KIS, SRLS, SRMS, KCS, LMCKS
      REAL*4 KCSNR, KCSRA, LIU, LCU, MCLU, LCS, MCLS, MIU, MCU
      REAL*4 MCS, KRD, LRD, ZTFSQ, ZFALB2, ZMAX, ZAL, ZALB
      REAL*4 EXZAL, EXZALB, BMC, BMIU, BMIS
      REAL*4 KIBN, KISN, LIUN, LISN, MIUN, MISN, DKION, DLION, KCBN,KCSN
      REAL*4 LMCKBN, LMCKSN, LCUN, MCLUN, LCSN, MCLSN, MCUN, MCSN, DCAPN
      REAL*4 TCAPN, DLI, DMI, DLC, DMC
      REAL*4 B(0:8)/ 0.,0., 0., 0., 0., .6, .3, 1., 1./
      REAL*4 KEX /2./,KMETA/100000./,LEX/1.5/,LMETA/10000./,LISSCR1/1./
      REAL*4 LISSCR2/4./,LIUSCR1/.4/,LIUSCR2/.5/,KISCR1/.6/,KISCR2/0./
      REAL*4 B3FAC/2./
      REAL*4 U1S, U1SS, U2S, U2P, U3S, U3D
      REAL*4 ZTFAC, DION, EN, Zp
      INTEGER*4 ISTATE, IMIN, J, I_GAS, I, IRC
C **********************************************************************
      REAL*4        A(0:172)
      COMMON / AA / A
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
      REAL*4          SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,  MIS
C **********************************************************************
      COMMON /BINPOL/ SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,  MIS
C **********************************************************************
      CHARACTER*3 CPRO, CTAR
      CHARACTER*5 CVERS
      REAL*4 AF, ZF, AT, ZT, DTARGET, EN0, DELT
      INTEGER*4 I_CHAR, I_OUTP, I_BRAN, IMAX, JQST(0:9), I_LOOP, QIN
      INTEGER*4 I_WRI
      COMMON /GLOB/
     #       AF, ZF, AT, ZT, DTARGET, EN0, DELT, I_WRI,
     #       I_CHAR, I_OUTP, I_BRAN, IMAX, JQST, I_LOOP, QIN,
     #       CPRO, CTAR, CVERS
C **********************************************************************
C
      IF( I_OUTP .EQ. 3) THEN
          WRITE(8,*) ' '
          WRITE(8,*) '<I>: EN = ', EN, ' MeV/u '
          WRITE(6,*) ' '
          WRITE(6,*) '<I>: EN = ', EN, ' MeV/u '
      ENDIF
 
      IF( EN .LE. 30. ) THEN
        WRITE(6,*) ' <I>: Energy = 30 MeV/u! Execution stopped! '
        IRC = -1
        GOTO 9999
      ENDIF
C
C *** ioniz. cross. calc. at projectile energy
C  ** data for ioniz. cross
      GA  = 1.+EN/931.5
      BSQ = 1.-1./GA/GA
      BETA  = SQRT(BSQ)
      BSA=18780.*BSQ
      CALL IONICRO(ZF, U1S, U1SS, U2S, U2P, U3S, U3D, IRC)
      IF( IRC .NE. 0 ) WRITE(6,*) '<E>: RC FROM IONICRO =', IRC
C
C *** capt. cross calc. at projectile energies
C  ** quantities used in non-radiative capture.
      D2=(GA-1.)/(GA+1.)
      D1=SQRT(D2)
      GAG=GA/(GA+1.)
      ETA=ALPH/BETA
      ZTFAC=ZT*ZT+ZT
C   * double ioniz. factor.
      DION=.00231*ZT*ZT/ZF/BETA
C
      DO 200, ISTATE=1, 6
C      calulate K,L,M ioniz. and capt. cross sections each unscreened
C      and screened
       GOTO (2340,2370,2400,2420,2440,2450) ISTATE
C ***  capt. into K proj. states from K,L,M target states
2340   KIB = ZTFAC * SKIB
       CALL CAPCROSS(ZF, ZT, SC, IRC)
       CALL RECCORK(ZF, U1S, SRKBEST, IRC)
       CALL RECCORLM(U2S, SRLU, SRMU, IRC)
       KCB = SC(1) + ZT * SRKBEST
       LMCKB = SC(2) + SC(3) + ZT * (SRLU + SRMU)
       IF( I_BRAN .EQ. 1 ) KIB = KIB / (1.+ZT*BP1S)**P1S
       KCBNR = SC(1)
       KCBRA = ZT * SRKBEST
       LCBNR = SC(2)
       LCBRA = ZT * SRLU
       MCBNR = SC(3)
C
2370   KIS = ZTFAC * SKIS
       ZP = ZF-.3
       CALL CAPCROSS(ZP, ZT, SC, IRC)
       CALL RECCORK(ZP, U1SS, SRKBEST, IRC)
       CALL RECCORLM(U2P, SRLS, SRMS, IRC)
       KCS = SC(1) + ZT * SRKBEST
       LMCKS = SC(2) + SC(3) + ZT * (SRLS + SRMS)
       IF( I_BRAN .EQ. 1 ) KIS = KIS / (1.+ZT*BP1SS)**P1SS
       KCSNR = SC(1)
       KCSRA = ZT * SRKBEST
C
C ***  capt. into L proj. states from K,L,M target states
2400   LIU = ZTFAC * (SU2S+3.*SU2P)/8.
       ZP = ZF - 1.7
       CALL CAPCROSS(ZP, ZT, SC, IRC)
       LCU = SC(2) + ZT * SRLU
       MCLU = SC(3) + ZT * SRMU
       IF( I_BRAN .EQ. 1 )
     #   LIU=ZTFAC*(SU2S/(1.+ZT*BP2SU)**P2SU+
     #       3.*SU2P/(1.+ZT*BP2PU)**P2PU)/8.
C
2420   LIS = ZTFAC*(SS2S+3.*SS2P)/8.
       ZP = ZF-3.
       CALL CAPCROSS(ZP, ZT, SC, IRC)
       LCS = SC(2)+ZT*SRLS
       MCLS = SC(3)+ZT*SRMU
       IF( I_BRAN .EQ. 1 )
     #     LIS=ZTFAC*(SS2S/(1.+ZT*BP2SS)**P2SS+
     #         3.*SS2P/(1.+ZT*BP2PS)**P2PS)/8.
C
C ***  capt. into M proj. states from K,L,M target states
2440   MIU = ZTFAC*SMIU
       ZP = ZF-8.
       CALL CAPCROSS(ZP, ZT, SC, IRC)
       MCU = SC(3)+ZT*SRMU
C
2450   MIS = ZTFAC * SMIS
       ZP = ZF - 19.
       CALL CAPCROSS(ZP, ZT, SC, IRC)
       MCS = SC(3) + ZT * SRMS
200   CONTINUE
C
C *** calc. of B factors
C
      KRD = (LIU-KIB)/(KEX*LIU+KMETA)
      LRD = (MIU-LIU)/(LEX*MIU+LMETA)
      ZTFSQ = ZT*ZT/ZF/ZF
      ZFALB2 = (ZF*ALPH/BETA)**2
      B(0) = (1.+LRD)/(1+LISSCR1*ZTFSQ*(LISSCR2+ZFALB2))
      B(1) = (1.+LRD)/(1+LIUSCR1*ZTFSQ*(LIUSCR2+ZFALB2))
      B(2) = (1.+KRD)/(1+KISCR1*ZTFSQ*(KISCR2+ZFALB2))
C
      IF( ZT .GT. ZF ) THEN
        ZMAX=ZT
      ELSE
        ZMAX=ZF
      ENDIF
      ZAL = ZMAX*ALPH
      ZALB = ZAL/BETA
      EXZAL = EXP(ZAL)
      EXZALB = EXP(ZALB)
      B(3) = (1+B3FAC*EXZALB) / (1+B3FAC*EXZAL)
      B(4) = 1.-KRD
      BMC = 1.-LRD
      BMIU = B(1)
      BMIS = B(0) * 0.8
      IF( J .LT. 10 ) THEN
        BMIU=0.
        BMIS=0.
      ENDIF
C
      IF( I_GAS .EQ. 1 ) THEN
C       gas target
        DO I = 0, 8
          B(I) = 1.
        ENDDO
        BMIS = 1.
        BMIU = 1.
        BMC = 1.
      ENDIF
C
C *** prepare final cross sections and A() parameters
C
      KIBN = KIB*B(2)
      KISN = KIS*B(2)
      LIUN = LIU*B(1)
      LISN = LIS*B(0)
      MIUN = MIU*BMIU
      MISN = MIS*BMIS
      DKION = DION*B(5)
      DLION = DION*B(6)
      KCBN = (KCBRA+KCBNR)*B(3)
      KCSN = (KCSRA+KCSNR)*B(3)
      LMCKBN = LMCKB*B(4)
      LMCKSN = LMCKS*B(4)
      LCUN = LCU*B(4)
      MCLUN = MCLU*BMC
      LCSN = LCS*B(4)
      MCLSN = MCLS*BMC
      MCUN = MCU*BMC
      MCSN = MCS*BMC
      DCAPN = KCBNR/900000.*B(7)
      TCAPN = 1.3*DCAPN*DCAPN*B(8)
C     differences per electron between screened and unscreened
      DLI = (LISN-LIUN)/7.
      DMI = (MISN-MIUN)/17.
      DLC = (LCUN-LCSN)/7.
      DMC = (MCUN-MCSN)/17.
C
      A(0) = 0.
      A(1) = 0.
      A(2) = 0.
      A(6) = 0.
      A(7) = 0.
      A(12) = 0.
      A(3) = -(1.+DCAPN+TCAPN)*(KCBN+LMCKBN)
      A(4) = KIBN
      A(10) =2.*KISN
      DO 300, I=2, 9
        A(6*I+4) = A(6*I-2)+LIUN+2*(I-2)*DLI
300   CONTINUE
      A(8) = KCBN+LMCKBN
      A(14)=KCSN/2.+LMCKSN
      A(9) = -A(4)-(1.+DCAPN+TCAPN)*A(14)
      DO 310 I=3, 10
        A(6*I+2) = (LCUN-(I-3)*DLC)*(11-I)/8+MCLUN
310   CONTINUE
      IF( J .LE. 10 )  GOTO 315
      DO 312 I=10, J-1
        A(6*I+4) = A(6*I-2)+MIUN+2.*(I-10)*DMI
312   CONTINUE
      DO 314 I=11, J
        A(6*I+2) = (MCUN-(I-11)*DMC)*(29-I)/18.
314   CONTINUE
315   DO 316 I=0, 1
        A(6*I+5) = DKION*A(6*I+10)*(I+1)/2.
316   CONTINUE
      DO 317 I=2, J-2
        A(6*I+5) = DLION*A(6*I+10)*(I+1)/2.
317   CONTINUE
C     readjustment for L and M shells
      DCAPN=DCAPN/2.
      TCAPN=TCAPN/2.
      DO 318 I=2, J
        A(6*I+1) = DCAPN*A(6*I-4)
318   CONTINUE
      DO 319 I=3, J
        A(6*I) = TCAPN*A(6*I-10)
319   CONTINUE
      DO 320 I=2, J-3
        A(6*I+3) = -A(6*I-2)-A(6*I-7)-A(6*I+8)*(1.+DCAPN+TCAPN)
320   CONTINUE
      A(6*J-9) = -A(6*J-14)-A(6*J-19)-A(6*J-4)*(1.+DCAPN)
      A(6*J-3)=-A(6*J-8)-A(6*J-13)-A(6*J+2)
      A(6*J+3)=-A(6*J-2)-A(6*J-7)
      IMIN = 6 * J+4
      DO 330 I=IMIN, 172
        A(I)=0.
330   CONTINUE
C
C ***  list cross sections
C
      IF( I_OUTP .EQ. 3 ) THEN
       WRITE(6,*)
       WRITE(6,*)' KIB        LIU        MIU        KCB        LCU  '//
     #   '      MCU        Double K ion.'
       WRITE(6,*)'                                  LMCKB      MCLU '//
     #    '                 Double L ion.'
       WRITE(6,*)' KIS        LIS        MIS        KCBNR      LCBNR'//
     #    '      MCBNR      Double capt.'
       WRITE(6,*)'                                  KCBRA      LCBRA'//
     #    '      TOTCAP     Triple capt. '
       WRITE(6,4001) KIBN,LIUN,MIUN,KCBN,LCUN,MCUN,DKION
       WRITE(6,4002) LMCKBN,MCLUN,DLION
       WRITE(6,4003) KISN,LISN,MISN,KCBNR*B(3),LCBNR*B(4),MCBNR*BMC,
     #              DCAPN*2.
       WRITE(6,4004) KCBRA*B(3),LCBRA*B(4),KCBN+LMCKBN,TCAPN*2.
       WRITE(6,*) ' '
       WRITE(6,*) ' Factors: '
       WRITE(6,4005) B(2), B(1), BMIU, B(3), B(4), BMC, B(5)
       WRITE(6,4006) B(4), BMC,  B(6)
       WRITE(6,4007) B(2), B(0), BMIS, B(3), B(4), BMC, B(7)
       WRITE(6,4008) B(3), B(4), B(8)
       WRITE(6,*)
C
       WRITE(8,*)
       WRITE(8,*)' KIB        LIU        MIU        KCB        LCU  '//
     #   '      MCU        Double K ion.'
       WRITE(8,*)'                                  LMCKB      MCLU '//
     #    '                 Double L ion.'
       WRITE(8,*)' KIS        LIS        MIS        KCBNR      LCBNR'//
     #    '      MCBNR      Double capt.'
       WRITE(8,*)'                                  KCBRA      LCBRA'//
     #    '      TOTCAP     Triple capt. '
       WRITE(8,4001) KIBN,LIUN,MIUN,KCBN,LCUN,MCUN,DKION
       WRITE(8,4002) LMCKBN,MCLUN,DLION
       WRITE(8,4003) KISN,LISN,MISN,KCBNR*B(3),LCBNR*B(4),MCBNR*BMC,
     #              DCAPN*2.
       WRITE(8,4004) KCBRA*B(3),LCBRA*B(4),KCBN+LMCKBN,TCAPN*2.
       WRITE(8,*) ' '
       WRITE(8,*) ' Factors: '
       WRITE(8,4005) B(2), B(1), BMIU, B(3), B(4), BMC, B(5)
       WRITE(8,4006) B(4), BMC,  B(6)
       WRITE(8,4007) B(2), B(0), BMIS, B(3), B(4), BMC, B(7)
       WRITE(8,4008) B(3), B(4), B(8)
       WRITE(8,*)
      ENDIF
4001  FORMAT(1X,7(G10.3, 1X))
4002  FORMAT(34X, 2(G10.3, 1X), 11X, G10.3)
4003  FORMAT(1X,7(G10.3, 1X))
4004  FORMAT(34X, 4(G10.3, 1X))
4005  FORMAT(7(F10.6, 1X))
4006  FORMAT(33X, 2(F10.6, 1X), 11X, F10.6)
4007  FORMAT(7(F10.6, 1X))
4008  FORMAT(33X, 2(F10.6, 1X), 11X, F10.6)
C
9999  RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE OUTPUT
C
C     WRITES SETUP ON FILE FOR FURTHER USE
C
      CHARACTER*3 CPRO, CTAR
C MFA
      CHARACTER*5 CVERS, COUTPUT
C MFE
C PCA
C     CHARACTER*8 COUTPUT
C     CHARACTER*5 CVERS
C PCE
C **********************************************************************
      INTEGER*4 I_CHAR, I_OUTP, I_BRAN, IMAX, JQST(0:9), I_LOOP, QIN
      INTEGER*4 I_WRI, I_WR
      REAL*4 AF, ZF, AT, ZT, DTARGET, EN0, DELT
      COMMON /GLOB/
     #       AF, ZF, AT, ZT, DTARGET, EN0, DELT, I_WRI,
     #       I_CHAR, I_OUTP, I_BRAN, IMAX, JQST, I_LOOP, QIN,
     #       CPRO, CTAR, CVERS
C **********************************************************************
C
      CALL PCHA('Enter OUTPUT file name', COUTPUT)
      OPEN( UNIT=1, FILE=COUTPUT//'.ginput', STATUS='UNKNOWN',ERR=999)
C PC  OPEN( UNIT=1, FILE=COUTPUT//'.gin', STATUS='UNKNOWN',ERR=999)
C
      WRITE(1,*) 'GLOBAL Input file:'
      WRITE(1,*)
     #  'Projectile:    A          Z        Q          Energy(MeV/u)'
      WRITE(1,11) NINT(AF), NINT(ZF), QIN, EN0
11    FORMAT(15x, I3, 8X, I2, 7X, I2, 11X, F7.1)
      WRITE(1,*)
     #  'Target:        A          Z    Dt(mg/cm^2) '
      WRITE(1,12) AT, NINT(ZT), DTARGET
12    FORMAT(12X, F6.1, 8X, I2, 3X, F8.1)
 
      IF( I_WRI .EQ. 10 ) THEN
        I_WR = 0
      ELSEIF ( I_WRI .EQ. 100 ) THEN
        I_WR = 1
      ELSEIF ( I_WRI .EQ. 1000 ) THEN
        I_WR = 2
      ELSEIF ( I_WRI .EQ. 10000 ) THEN
        I_WR = 3
      ELSE
        I_WR = 2
      ENDIF
 
      WRITE(1,*)
     #  'Options:     I_CHAR    I_LOOP   I_OUTP     I_WR'
      WRITE(1,13) I_CHAR, I_LOOP, I_OUTP, I_WR
13    FORMAT(16x, I1, 10X, I1, 8X, I1, 9X, I1)
      WRITE(1,*) 'Q-states: '
      WRITE(1,14) JQST
14    FORMAT(3x, I2, 1X, I2, 1X, I2, 1X, I2, 1X, I2, 1X, I2, 1X, I2,
     #       1X, I2, 1X, I2, 1X, I2)
      WRITE(1,*)
      WRITE(1,*)
      WRITE(1,*) 'Comments:'
      WRITE(1,*)
     #    'Options: I_CHAR: =0  ====> charge state at target exit'
      WRITE(1,*) '                 =1  ====> equilibrium charge state'
      WRITE(1,*) '                 =2  ====> charge-state evolution'
      WRITE(1,*) '         I_LOOP: =0  ====> no loop'
      WRITE(1,*) '                 =1  ====> loop over Z(projectile)'
      WRITE(1,*) '                 =2  ====>           incident energy'
      WRITE(1,*) '                 =3  ====>           incident Q state'
      WRITE(1,*) '                 =4  ====>           Z(target)'
      WRITE(1,*) '                 =5  ====>           D(target)'
      WRITE(1,*) '         I_OUTP: =0  ====> output on screen'
      WRITE(1,*) '                 =1  ====>        on file'
      WRITE(1,*) '                 =2  ====>        on screen/file'
      WRITE(1,*) '         I_WR:   =0,1,2,3 '//
     # 'multiplies output steps by 10,100,1000,10000'
      WRITE(1,*) '                 reasonable values: I_WR=1,2,3'
C
      CLOSE(1)
      RETURN
 
999   CONTINUE
      WRITE(6,*) '<W>: File not written'
      RETURN
      END
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE INPUT
C
C     READS SETUP FROM FILE
C
      CHARACTER*(80) C_REAL, C_SUB(30)
      CHARACTER*3 CPRO, CTAR
      CHARACTER*5 CVERS
C **********************************************************************
      INTEGER*4 I_CHAR, I_OUTP, I_BRAN, IMAX, JQST(0:9), I_LOOP, QIN
      INTEGER*4 NDUM(30), NSUB, I_WRI, I, I_WR, len
      REAL*4 AF, ZF, AT, ZT, DTARGET, EN0, DELT
C
      COMMON /GLOB/
     #       AF, ZF, AT, ZT, DTARGET, EN0, DELT, I_WRI,
     #       I_CHAR, I_OUTP, I_BRAN, IMAX, JQST, I_LOOP, QIN,
     #       CPRO, CTAR, CVERS
C **********************************************************************
C MFA
      CHARACTER*5 CINPUT
      CHARACTER*15 CFILE
C MFE
C PCA
C     CHARACTER*8 CINPUT
C PCE
      COMMON / OUTIN/ CINPUT
C **********************************************************************
C
C MFA
      CALL PCHA('Enter INPUT file name or <RETURN> for Menue', CINPUT)
      len = index(cinput,' ')-1
      CFILE = CINPUT(1:len)//'.ginput'
      OPEN( UNIT=1, FILE=CFILE, STATUS='OLD',ERR=999)
C MFE
C PCA
C     CALL PCHA('Enter INPUT file name or <ANY LETTER> for Menue',
C    #     CINPUT)
C     OPEN( UNIT=1, FILE=CINPUT//'.gin', STATUS='OLD',ERR=999)
C PCE
C
      READ(1,*)
 
C *** Projectile
      READ(1,*)
      READ(1, FMT='(A)', ERR=90, END=90) C_REAL
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_SSTR( C_REAL, C_SUB, NDUM, NSUB )
        IF( NSUB .NE. 4 ) THEN
          WRITE(6,*) '<E>: Error in INPUT file'
          STOP
        ENDIF
        CALL C_CHRE( C_SUB(1), AF )
        CALL C_CHRE( C_SUB(2), ZF )
        CALL C_CHIN( C_SUB(3), QIN )
        CALL C_CHRE( C_SUB(4), EN0 )
      ELSE
        WRITE(6,*) '<E>: Error in INPUT file'
        STOP
      ENDIF
      IF( ZF .GT. 96. ) THEN
         WRITE(6,*) '<W>: ZF > 96! ZF set to 96!'
         ZF = 96.
      ENDIF
      IF( QIN .GT. 28 ) THEN
         WRITE(6,*) '<W>: QIN > 28! QIN set to 28!'
         QIN = 28
      ENDIF
      IF( QIN .GE. ZF ) THEN
         WRITE(6,*) '<W>: QIN >= ZF! QIN set to 0!'
         QIN = 0
      ENDIF
         
C
C *** Target
      READ(1,*)
      READ(1, FMT='(A)', ERR=90, END=90) C_REAL
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_SSTR( C_REAL, C_SUB, NDUM, NSUB )
        IF( NSUB .NE. 3 ) THEN
          WRITE(6,*) '<E>: Error in INPUT file'
          STOP
        ENDIF
        CALL C_CHRE( C_SUB(1), AT )
        CALL C_CHRE( C_SUB(2), ZT )
        CALL C_CHRE( C_SUB(3), DTARGET )
      ELSE
        WRITE(6,*) '<E>: Error in INPUT file'
        STOP
      ENDIF
      IF( ZT .GT. 96. ) THEN
         WRITE(6,*) '<W>: ZT > 96! ZT set to 96!'
         ZT = 96.
      ENDIF
C
C *** Options, Loop, Output
      READ(1,*)
      READ(1, FMT='(A)', ERR=90, END=90) C_REAL
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_SSTR( C_REAL, C_SUB, NDUM, NSUB )
        IF( NSUB .NE. 4 ) THEN
          WRITE(6,*) '<E>: Error in INPUT file'
          STOP
        ENDIF
        CALL C_CHIN( C_SUB(1), I_CHAR )
        CALL C_CHIN( C_SUB(2), I_LOOP )
        CALL C_CHIN( C_SUB(3), I_OUTP )
        CALL C_CHIN( C_SUB(4), I_WR )
        IF( I_WR .EQ. 0 ) THEN
          I_WRI = 10
        ELSEIF ( I_WR .EQ. 1 ) THEN
          I_WRI = 100
        ELSEIF ( I_WR .EQ. 2 ) THEN
          I_WRI = 1000
        ELSEIF ( I_WR .EQ. 3 ) THEN
          I_WRI = 10000
        ELSE
          I_WRI = 1000
        ENDIF
      ELSE
        WRITE(6,*) '<E>: Error in INPUT file'
        STOP
      ENDIF
C
C *** Qstates
      READ(1,*,ERR=90, END=80)
      READ(1, FMT='(A)', ERR=90, END=80) C_REAL
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_SSTR( C_REAL, C_SUB, NDUM, NSUB )
        DO I = 0, MIN(9,NSUB-1)
          CALL C_CHIN( C_SUB(I+1), JQST(I) )
        ENDDO
        IMAX = MIN(10,NSUB)
      ELSE
        DO I = 0, 9
          JQST(I) = I
        ENDDO
        IMAX = 10
      ENDIF
      CLOSE(1)
      RETURN
C
80    DO I = 0, 9
         JQST(I) = I
      ENDDO
      IMAX = 10
      CLOSE(1)
      RETURN
C
90    STOP
999   CONTINUE
C MFA
      CINPUT = 'test0'
C MFE
C PCA
C     CINPUT = 'test0000'
C PCE
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE MENUE
C
C     PRODUCES MENUE FOR PARAMETER CHANGES
C
C     CHARACTER*1 CBRANDT/'N'/
      CHARACTER*3 CPRO, CTAR
      CHARACTER*5 CVERS
      CHARACTER*8 CI
C **********************************************************************
      REAL*4 AF, ZF, AT, ZT, DTARGET, EN0, DELT, AKEEP
      INTEGER*4 I_CHAR, I_OUTP, I_BRAN, IMAX, JQST(0:9), I_LOOP, QIN
      INTEGER*4 IPAR, I_WRI, I_WR, I, IK, IMAXOLD, IRC, K
C
      COMMON /GLOB/
     #       AF, ZF, AT, ZT, DTARGET, EN0, DELT, I_WRI,
     #       I_CHAR, I_OUTP, I_BRAN, IMAX, JQST, I_LOOP, QIN,
     #       CPRO, CTAR, CVERS
C **********************************************************************
C MFA
      CHARACTER*5 CINPUT, CFILE/'test0'/
C MFE
C PCA
C     CHARACTER*8 CINPUT, CFILE/'test0000'/
C PCE
      COMMON / OUTIN/ CINPUT
C **********************************************************************
      CVERS = '02/97'
C
      IF( AF .EQ. 0. .AND. ZF .EQ. 0. .AND.
     #    AT .EQ. 0. .AND. ZT .EQ. 0. ) THEN
         AF = 238.
         ZF = 92.
         AT = 63.5
         ZT = 29.
         DTARGET = 100.
         EN0 = 430.
         I_WR = 1
         I_CHAR = 0
         I_OUTP = 2
         I_LOOP = 0
         QIN = 2
         DO I = 0, 9
           JQST(I) = I
         ENDDO
      ENDIF
C
1     CONTINUE
      IF( QIN .GT. 28 ) THEN
        WRITE(6,*) '<W>: QIN > 28! QIN set to 28!'
        QIN = 28
      ENDIF
      IF( EN0 .LT. 0. ) THEN 
        WRITE(6,*) '<W>: EN0 = 0! EN0 set to 100!'
        EN0 = 100.
      ENDIF
      IF( DTARGET .LT. 0. ) THEN 
        WRITE(6,*) '<W>: DTARGET = 0! DTARGET set to 100!'
        DTARGET = 100.
      ENDIF
C
      WRITE(6,*)
      WRITE(6,*)
      WRITE(6,*)
      WRITE(6,*)
     #' ******************* GLOBAL:  Q-STATE CALCULATIONS ********'
     #//'****** Version ',CVERS
      WRITE(6,*)
      WRITE(6,11) ZF, AF, QIN, EN0
      WRITE(6,12) ZT, AT, DTARGET
C
11    FORMAT(' Projectile: Z = ', F3.0, '  (1)  A =', F5.0,
     #      '  (2)    Qe = ', I2, '  (3)    E = ', F6.1, ' MeV/u (4)')
12    FORMAT(' Target:     Z = ', F4.1, ' (5)  A = ', F5.1,
     #      ' (6)    D (target) = ', G9.4, ' mg/cm^2   (7)' )
C
      WRITE(6,*) ' '
      WRITE(6,*) ' Options: Q-state at target exit:       ' //
     #            '       ===> 0 '
      WRITE(6,*) '          equilibrium Q-states:         ' //
     #            '       ===> 1 '
      WRITE(6,14) I_CHAR
14    FORMAT( '           evolution of Q-states:       ',
     #        '        ===> 2:   ', I1, 18x, '(8)')
C
      WRITE(6,*) ' '
      WRITE(6,*) ' Loop:    no loop             ===> 0'
     #           // '       over inc. Q-state  ===> 3'
      WRITE(6,*) '          over Z(projectile)  ===> 1'
     #           // '       over Z(target)     ===> 4'
      WRITE(6,15) I_LOOP
15    FORMAT('           over inc. energy    ===> 2',
     #          '       over D(target)     ===> 5:    ' ,I1,'  (9)')
C
      WRITE(6,*) ' '
      WRITE(6,*) ' Output:  on screen:          ===> 0 '
     #          // '      Freq. of Output: 1/10    ===> 0 '
      WRITE(6,*) '          on file:            ===> 1 '
     #          // '      (Option 2 only)  1/100   ===> 1 '
      WRITE(6,*) '          on screen/file:     ===> 2 '
     #          // '                       1/1000  ===> 2 '
      WRITE(6,*) '          of cross sections:  ===> 3:'
     #          // '                       1/10000 ===> 3:'
      IF( I_WRI .EQ. 1 ) THEN
        I_WR = 0
      ELSEIF ( I_WRI .EQ. 100 ) THEN
        I_WR = 1
      ELSEIF ( I_WRI .EQ. 1000 ) THEN
        I_WR = 2
      ELSEIF ( I_WRI .EQ. 10000 ) THEN
        I_WR = 3
      ELSE
        I_WR = 2
      ENDIF
      WRITE(6,16) I_OUTP, I_WR
16    FORMAT( 36X, I1,'   (10)', 30X, I1, ' (11)')
C
      IF( IMAX .EQ. 0 ) THEN
        DO I = 0, 9
          JQST(I) = I
        ENDDO
        IMAX = 10
      ENDIF
C
      WRITE(6,*)
      IF( IMAX .EQ. 1 ) WRITE(6,171) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 2 ) WRITE(6,172) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 3 ) WRITE(6,173) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 4 ) WRITE(6,174) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 5 ) WRITE(6,175) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 6 ) WRITE(6,176) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 7 ) WRITE(6,177) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 8 ) WRITE(6,178) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 9 ) WRITE(6,179) (JQST(I), I=0,IMAX-1 )
      IF( IMAX .EQ. 10) WRITE(6,180) (JQST(I), I=0,IMAX-1 )
171   FORMAT(' Q-states to be printed (max. 10):    ',
     #       (1X, I2), 32x, '   (12)')
172   FORMAT(' Q-states to be printed (max. 10):    ',
     #      2(1X, I2), 29x, '   (12)')
173   FORMAT(' Q-states to be printed (max. 10):    ',
     #      3(1X, I2), 26x, '   (12)')
174   FORMAT(' Q-states to be printed (max. 10):    ',
     #      4(1X, I2), 23x, '   (12)')
175   FORMAT(' Q-states to be printed (max. 10):    ',
     #      5(1X, I2), 20x, '   (12)')
176   FORMAT(' Q-states to be printed (max. 10):    ',
     #      6(1X, I2), 17x, '   (12)')
177   FORMAT(' Q-states to be printed (max. 10):    ',
     #      7(1X, I2), 14x, '   (12)')
178   FORMAT(' Q-states to be printed (max. 10):    ',
     #      8(1X, I2), 11x, '   (12)')
179   FORMAT(' Q-states to be printed (max. 10):    ',
     #      9(1X, I2),  8x, '   (12)')
180   FORMAT(' Q-states to be printed (max. 10):    ',
     #     10(1X, I2),  5x, '   (12)')
      WRITE(6,*)
C
      WRITE(6,*) '       '
      IPAR = 0
      CALL PILO(
     #'Number of the parameter to be changed ( 0 to start/ -1 to exit):'
     #   , IPAR)
C
      IF( IPAR .LT. 0 ) THEN
        STOP
      ENDIF
C
      IF( IPAR .EQ. 0 ) GOTO 900
C
101   IF( IPAR .EQ. 1 ) THEN
        CALL C_RECH( ZF, CPRO)
        CALL PCHA('Enter new Z(projectile) or symbol', CPRO )
C
        DO 1001 IK = 0, 9
          CALL C_INCH( IK , CI )
          IF( CI .EQ. CPRO(1:1) ) THEN
            CALL C_CHRE( CPRO, ZF )
            IF( ZF .GT. 96. ) THEN
              WRITE(6,*) '<E>: ZF > 96! ZF set to 96!'
              ZF = 96.
            ENDIF
            CALL ELEMENT( ZF, AF, CPRO, 3, IRC )
            IF( IRC .EQ. -1 ) GOTO 101
            GOTO 1101
          ENDIF
1001    CONTINUE
C
        CALL ELEMENT( ZF, AF, CPRO, 4, IRC )
        IF( IRC .EQ. -1 ) GOTO 101
C
1101    IF( ZF .LE. 28. ) THEN
          WRITE(6,*) '<W>: ZP < 29! GLOBAL is designed for ZP > 28!'
        ENDIF
C
        CALL PRSO('Enter new A(projectile)', AF, 3 )
        IF( ZF .GT. AF ) THEN
           WRITE(6,*) ' <E>: ZF > AF ! '
           GOTO 101
        ENDIF
        IF( NINT(ZF) .LE. QIN ) GOTO 1103
      ENDIF
C
102   IF( IPAR .EQ. 2 ) THEN
        CALL PRSO('Enter new A(projectile)', AF, 3)
        CALL ELEMENT( ZF, AF, CPRO, 5, IRC )
        CALL PCHA('Enter new Z(projectile) or symbol', CPRO )
C
        DO 1002 IK = 0, 9
          CALL C_INCH( IK , CI )
          IF( CI .EQ. CPRO(1:1) ) THEN
            CALL C_CHRE( CPRO, ZF )
            IF( ZF .GT. 96. ) THEN
              WRITE(6,*) '<E>: ZF > 96! ZF set to 96!'
              ZF = 96.
            ENDIF
            GOTO 1102
          ENDIF
1002    CONTINUE
C
        AKEEP = AF
        CALL ELEMENT( ZF, AF, CPRO, 4, IRC )
        AF = AKEEP
        IF( IRC .EQ. -1 ) GOTO 102        
C
1102    IF( NINT(ZF) .LE. QIN ) GOTO 1103
      ENDIF
C
103   IF( IPAR .EQ. 3 ) THEN
1103    CALL PILO('Enter new Q state', QIN )
        IF( QIN .LT. 0 ) GOTO 103
        IF( QIN .GT. 28. ) THEN
           WRITE(6,*) '<E>: QIN > 28! QIN set to 28!'
           QIN = 28.
        ENDIF
        IF( QIN .GE. NINT(ZF) ) THEN
           WRITE(6,*) '<W>: QIN >= ZF! QIN set to 0!'
           QIN = 0
        ENDIF
      ENDIF
C
104   IF( IPAR .EQ. 4 ) THEN
        CALL PRSO('Enter new incident energy', EN0, 4)
        IF( EN0 .LE. 0. ) GOTO 104
        IF( EN0 .GT. 2000. ) THEN
          WRITE(6,*) '<I>: EN0(max) = 2000.! Use EN0 <= 2000.'
          GOTO 104
        ENDIF
      ENDIF
C
105   IF( IPAR .EQ. 5 ) THEN
        CALL C_RECH( ZT, CTAR)
        CALL PCHA('Enter new Z(target) or symbol', CTAR )
C
        DO 1005 IK = 0, 9
          CALL C_INCH( IK , CI )
          IF( CI .EQ. CTAR(1:1) ) THEN
            CALL C_CHRE( CTAR, ZT )
            IF( ZT .GT. 96. ) THEN
              WRITE(6,*) '<E>: ZT > 96! ZT set to 96!'
              ZT = 96.
            ENDIF
            CALL ELEMENT( ZT, AT, CTAR, 1, IRC )
            IF( IRC .EQ. -1 ) GOTO 105
            GOTO 1105
          ENDIF
1005    CONTINUE
C
        CALL ELEMENT( ZT, AT, CTAR, 2, IRC )
        IF( IRC .EQ. -1 ) GOTO 105
C
1105    CALL PRSO('Enter new A(target)', AT, 3 )
        IF( ZT .GT. AT ) THEN
          WRITE(6,*) ' <E>: ZT > AT ! '
          GOTO 105
        ENDIF
      ENDIF
C  
106   IF( IPAR .EQ. 6 ) THEN
        CALL PRSO('Enter new A(target)', AT, 3)
        CALL ELEMENT( ZT, AT, CTAR, 5, IRC )
        CALL PCHA('Enter new Z(target) or symbol', CTAR )
C
        DO 1006 IK = 0, 9
          CALL C_INCH( IK , CI )
          IF( CI .EQ. CTAR(1:1) ) THEN
            CALL C_CHRE( CTAR, ZT )
            IF( ZT .GT. 96. ) THEN
              WRITE(6,*) '<E>: ZT > 96! ZT set to 96!'
              ZT = 96.
            ENDIF
C            CALL ELEMENT( ZT, At, CTAR, 1, IRC )
C            IF( IRC .EQ. -1 ) GOTO 106
            GOTO 1106
          ENDIF
1006    CONTINUE
C
        AKEEP = AT        
        CALL ELEMENT( ZT, AT, CTAR, 4, IRC )
        AT = AKEEP
        IF( IRC .EQ. -1 ) GOTO 106
      ENDIF
C
1106  CONTINUE
107   IF( IPAR .EQ. 7 ) THEN
        CALL PRSO('Enter new target thickness', DTARGET, 4)
        IF( DTARGET .EQ. 0. ) GOTO 107
      ENDIF
C
109   IF( IPAR .EQ. 8 ) THEN
         CALL PILO('Enter new option', I_CHAR)
         IF( I_CHAR .LT. 0 .OR. I_CHAR .GT. 2 ) I_CHAR = 0
      ENDIF
C
110   IF( IPAR .EQ. 9 ) THEN
        CALL PILO('Enter new loop option', I_LOOP )
        IF( I_LOOP .LT. 0 .OR. I_LOOP .GT. 5 ) I_LOOP = 0
      ENDIF
C
111   IF( IPAR .EQ. 10 ) THEN
         CALL PILO('Enter new output option', I_OUTP )
         IF( I_OUTP .LT. 0 .OR. I_OUTP .GT. 3 ) I_OUTP = 0
      ENDIF
C
112   IF( IPAR .EQ. 11 ) THEN
         CALL PILO('Enter new output frequency', I_WR )
         IF( I_WR .LT. 0 .OR. I_WR .GT. 3 ) I_WR = 2
         IF( I_WR .EQ. 0 ) THEN
           I_WRI = 10
         ELSEIF ( I_WR .EQ. 1 ) THEN
           I_WRI = 100
         ELSEIF ( I_WR .EQ. 2 ) THEN
           I_WRI = 1000
         ELSEIF ( I_WR .EQ. 3 ) THEN
           I_WRI = 10000
         ELSE
           I_WRI = 1000
         ENDIF
      ENDIF
C
113   IF( IPAR .EQ. 12 ) THEN
        IMAXOLD = IMAX
        CALL PILO('Enter number of Q-states to be printed', IMAX)
        IF( IMAX .GT. 10 ) THEN
          WRITE(6,*) '<I>: Maximum number of Q-states equals 10!'
          IMAX = 10
        ENDIF
        DO I = IMAXOLD, IMAX - 1
          DO IK = 0, 9
           DO K = 0, I-1
             IF( JQST(K) .EQ. IK ) THEN
               GOTO 1131
             ENDIF
           ENDDO
           JQST(I) = IK
           GOTO 1132
1131      ENDDO
1132    ENDDO
        DO I = IMAX, 9
           JQST(I) = -1
        ENDDO
        CALL PILO10('Enter Q-state to be printed', JQST, IMAX )
      ENDIF
C
      GOTO 1
C
900   CONTINUE
C
      IF( I_WR .EQ. 0 ) THEN
        I_WRI = 10
      ELSEIF ( I_WR .EQ. 1 ) THEN
        I_WRI = 100
      ELSEIF ( I_WR .EQ. 2 ) THEN
        I_WRI = 1000
      ELSEIF ( I_WR .EQ. 3 ) THEN
        I_WRI = 10000
      ELSE
        I_WRI = 1000
      ENDIF
C
      IF( FLOAT(INT(ZF)) .NE. ZF   ) THEN
         WRITE(6,*)
         WRITE(6,*) '<E>: Composites not possible as beams!!'
         WRITE(6,*)
         GOTO 101
      ENDIF
C
      IF( ZF .LT. 29. ) THEN
        WRITE(6,*)
        WRITE(6,*) '<W>: GLOBAL is designed for ZF > 28!'
        WRITE(6,*)
      ENDIF
C
C     IF( I_BRAN .EQ. 0 ) CBRANDT = 'N'
C     IF( I_BRAN .EQ. 1 ) CBRANDT = 'Y'
C     CALL PYES(
C    # 'Use binding-polarization factor for x-sections (y/n)', CBRANDT)
C     IF( CBRANDT .NE. 'N' .AND. CBRANDT .NE. 'n' ) I_BRAN = 1
C
      I_BRAN = 1
C
      IF( I_OUTP .EQ. 1 .OR. I_OUTP .EQ. 2 .OR. I_OUTP .EQ. 3) THEN
C ***   OPEN OUTPUT DATASET
        CFILE = CINPUT
        CALL PCHA('Enter output dataset name', CFILE)
C MFA
C       OPEN( UNIT=8, FILE='/s5/blank/qstate/data/'//CFILE//'.globout',
C    #        STATUS='UNKNOWN')
        OPEN( UNIT=8, FILE=CFILE//'.globout', STATUS='UNKNOWN')
C MFE
C PCA
C       OPEN( UNIT=8, FILE=CFILE//'.out', STATUS='UNKNOWN')
C PCE
      ENDIF
 
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      FUNCTION FNPOL(X)
C
C     needed for POLARIZATION correction
C
      REAL*4 FNPOL, X
C
C **********************************************************************
      FNPOL =
     #      EXP(-2.*X)/(.031+.21*SQRT(X)+.005*X-.069*X*SQRT(X)+.324*X*X)
      RETURN
      END
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE IONICRO(ZF, U1S, U1SS, U2S, U2P, U3S, U3D, IRC)
C
C     subroutine for ionization  cross sections / (ZT^2+ZT)
C
      CHARACTER*3 CFILE
      INTEGER*4 I, IA, IC, IN, IRC, JA
      REAL*4 ZF, U1S, U1SS, U2S, U2P, U3S, U3D
      REAL*4 THETAK(4), THETAL(4), THETAM(4), ETATHKL(10), ETATHM(40)
      REAL*4 ETATH(40), THETA(4), KIS, ETH1S, ETH1SS, ETH2P, ETH2S
      REAL*4 ETH3D, ETH3S, PP, REL, RELK, RELKS, RTR, SI, SS3D, SS3P
      REAL*4 SS3S, SU3D, SU3P, SU3S, TH1S, TH1SS, TH2P, TH2S, TH3D, TH3S
      REAL*4 TR1S, TR1SS, TR2PS, TR2PU, TR2SS, TR2SU, TR3SS, TR3SU
      REAL*4 CX, X, Z1S, Z2P, Z2S, Z3D, Z3S
      COMMON /ETATH/  ETATH, THETA
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
      REAL*4          SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,  MIS
C
      COMMON /BINPOL/ SKIB, BP1S,  P1S,  SKIS, BP1SS, P1SS,
     #                SU2S, BP2SU, P2SU, SU2P, BP2PU, P2PU,
     #                SS2S, BP2SS, P2SS, SS2P, BP2PS, P2PS,
     #                SMIU, SMIS,  KIB,  KCB,  LIS,  MIS
C **********************************************************************
C
C     THETA K in F tables of Benka and Kropf, ADND tables 22 (1978) 219
      DATA THETAK /.5334,.7113,.9486,1.2649 /
C     THETA L in F tables of Benka and Kropf, ADND tables 22 (1978) 219
      DATA THETAL /.4743,.6325,.8434,1.1247 /
C     THETA M in F tables of Johnson et al., ADND tables 24 (1979) 1
      DATA THETAM /.3,.4,.5,.6 /
C     ETA/THSQ K and L in F tables of Benka and Kropf
      DATA ETATHKL/.01,.01334,.01778,.02371,.03162,.04217,.05623,
     #           .06494,.07499,.0866/
C     ETA/THSQ M in F tables of Johnson et al.
      DATA ETATHM    /.04,.045,.05,.06,.07,.08,.1,.15,.2,.3,.4,.5,.6,
     #                .7,.8,.9,1.,1.3,1.5,1.7,2.,2.5,3.,3.5,4.,4.5,5.,
     #                5.5,6.,6.5,7.,8.,10.,13.,15.,20.,30.,45.,70.,100./
C **********************************************************************
      Z1S=ZF-.3
      Z2S=ZF-1.7
      Z2P=ZF-4.15
      Z3S=ZF-8.8
      Z3D=ZF-21.5
C
      IF( ZF .LE. 0. .OR. U1S .LE. 0. ) THEN
        TH1S = 0.
        ETH1S = 0.
      ELSE
        TH1S=U1S/U0/ZF/ZF
        ETH1S=BSA/(ZF*TH1S)**2
      ENDIF
      IF( Z1S .LE. 0. .OR. U1SS .LE. 0. ) THEN
        TH1SS = 0.
        ETH1SS = 0.
      ELSE
        TH1SS=U1SS/U0/Z1S/Z1S
        ETH1SS=BSA/(Z1S*TH1SS)**2
      ENDIF
      IF( Z2S .LE. 0. .OR. U2S .LE. 0. ) THEN
        TH2S = 0.
        ETH2S = 0.
      ELSE
        TH2S=U2S/U0/Z2S/Z2S*4.
        ETH2S=BSA/(Z2S*TH2S)**2
      ENDIF
      IF( Z2P .LE. 0. .OR. U2P .LE. 0. ) THEN
        TH2P = 0.
        ETH2P = 0.
      ELSE
        TH2P=U2P/U0/Z2P/Z2P*4.
        ETH2P=BSA/(Z2P*TH2P)**2
      ENDIF
      IF( Z3S .LE. 0. .OR. U3S .LE. 0. ) THEN
        TH3S = 0.
        ETH3S = 0.
      ELSE
        TH3S=U3S/U0/Z3S/Z3S*9.
        ETH3S=BSA/(Z3S*TH3S)**2
      ENDIF
      IF( Z3D .LE. 0. .OR. U3D .LE. 0. ) THEN
        TH3D = 0.
        ETH3D = 0.
      ELSE
        TH3D=U3D/U0/Z3D/Z3D*9.
        ETH3D=BSA/(Z3D*TH3D)**2
      ENDIF
C
C *** Compute ioniz. cross section
10    CONTINUE
      DO 11 I = 1, 10
        ETATH(I) = ETATHKL(I)
11    CONTINUE
      DO 12 I = 1, 4
        THETA(I) = THETAK(I)
12    CONTINUE
C
      DO 13 IN = 1, 10
        IC=10+IN
        ETATH(IC)=10.*ETATH(IN)
        JA=20+IN
        ETATH(JA)=100.*ETATH(IN)
        IA=30+IN
        ETATH(IA)=1000.*ETATH(IN)
13    CONTINUE
C
C     unscreened 1s cross section
100   CONTINUE
      CFILE = 'fk0'
      CALL INTPOKLM(ETH1S, TH1S, ZF,  U1S, CFILE, SI,   TR1S,  PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -1
         SKIB = 0.
         P1S = 0.
         BP1S = 0.
         GOTO 200
      ENDIF
C     per electron and per (ZT^2+ZT),contains transverse
      KIB=SI/2.
C     sq. root of 1/relativity corr.
      REL=1.+.0000133*BETA*ZF*ZF
      RELK=1./REL/REL
C     X and CX needed for binding-polarization correction
      SKIB=KIB*RELK
      P1S=PP-1.
      X=2.*SQRT(ETH1S)
      CX=1.5/X
      CALL BIPOCO1S( X, CX, ETH1S,  TH1S,  ZF,  BP1S  )
C
200   CONTINUE
C     screened 1s cross section
      CFILE = 'fk0'
      CALL INTPOKLM(ETH1SS,TH1SS,Z1S, U1SS,CFILE, SI,   TR1SS ,PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -2
         SKIS = 0.
         P1SS = 0.
         BP1SS = 0.
         GOTO 300
      ENDIF
C     per electron and per (ZT^2+ZT),contains transverse
      KIS=SI/2.
C     sq. root of 1/relativity corr.
      REL=1.+.0000133*BETA*Z1S*Z1S
      RELKS=1./REL/REL
      SKIS=KIS*RELKS
      P1SS=PP-1.
C     X and CX needed for binding-polarization correction
      X=2.*SQRT(ETH1SS)
      CX=1.5/X
      CALL BIPOCO1S( X, CX, ETH1SS, TH1SS, Z1S, BP1SS )
C
300   CONTINUE
C *** Compute unscreened 2s and 2p cross sections from 2s data
      DO 301 I = 1, 4
        THETA(I) = THETAL(I)
301   CONTINUE
      CFILE = 'fl1'
      CALL INTPOKLM(ETH2S, TH2S, Z2S, U2S, CFILE, SU2S, TR2SU, PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -3
         SU2S = 0.
         P2SU = 0.
         BP2SU = 0.
         GOTO 400
      ENDIF
      P2SU=PP-1.
      X=4.*SQRT(ETH2S)
      CX=3./X
      CALL BIPOCO2S( X, CX, ETH2S,  TH2S,  Z2S, BP2SU )
C
400   CONTINUE
      CFILE = 'fl2'
      CALL INTPOKLM(ETH2S, TH2S, Z2S, U2S, CFILE, SU2P, TR2PU, PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -4
         SU2P = 0.
         P2PU = 0.
         BP2PU = 0.
         GOTO 500
      ENDIF
      P2PU=PP-1.
      X=4.*SQRT(ETH2S)
      CX=2.5/X
C     no relativity correction for L shell
      CALL BIPOCO2P( X, CX, ETH2S,  TH2S,  Z2S, BP2PU )
C
500   CONTINUE
C *** Compute screened 2s and 2p cross sections from 2p data.
      CFILE = 'fl1'
      CALL INTPOKLM(ETH2P, TH2P, Z2P, U2P, CFILE, SS2S, TR2SS, PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -5
         SS2S = 0.
         P2SS = 0.
         BP2SS = 0.
         GOTO 600
      ENDIF
      P2SS=PP-1.
      X=4.*SQRT(ETH2P)
      CX=3./X
      CALL BIPOCO2S( X, CX, ETH2P,  TH2P,  Z2P, BP2SS )
C
600   CONTINUE
      CFILE = 'fl2'
      CALL INTPOKLM(ETH2P, TH2P, Z2P, U2P, CFILE, SS2P, TR2PS, PP, IRC )
      IF (IRC .NE. 0 ) THEN
         IRC = -6
         SS2P = 0.
         P2PS = 0.
         BP2PS = 0.
         GOTO 700
      ENDIF
      P2PS=PP-1.
      X=4.*SQRT(ETH2P)
      CX=2.5/X
      CALL BIPOCO2P( X, CX, ETH2P,  TH2P,  Z2P, BP2PS )
C
700   CONTINUE
C     Compute unscreened 3s,3p,3d cross sections from 3s data
      DO 701 I = 1, 4
        THETA(I) = THETAM(I)
701   CONTINUE
      DO 702 I = 1, 40
        ETATH(I) = ETATHM(I)
702   CONTINUE
      CFILE = 'fm1'
      CALL INTPOKLM(ETH3S, TH3S, Z3S, U3S, CFILE, SU3S, TR3SU, PP, IRC )
      CFILE = 'fm2'
      CALL INTPOKLM(ETH3S, TH3S, Z3S, U3S, CFILE, SU3P, RTR,   PP, IRC )
      CFILE = 'fm3'
      CALL INTPOKLM(ETH3S, TH3S, Z3S, U3S, CFILE, SU3D, RTR,   PP, IRC )
      IF( IRC .NE. 0 ) IRC = -7
C     per electron
      SMIU=(SU3S+SU3P+SU3D)/18.
      IF( SMIU .EQ. 0. ) SMIU = 1.E10
C
800   CONTINUE
C *** Compute screened 3s,3p,3d cross sections from 3d data
      CFILE = 'fm1'
      CALL INTPOKLM(ETH3D, TH3D, Z3D, U3D, CFILE, SS3S, TR3SS, PP, IRC )
      CFILE = 'fm2'
      CALL INTPOKLM(ETH3D, TH3D, Z3D, U3D, CFILE, SS3P, RTR,   PP, IRC )
      CFILE = 'fm3'
      CALL INTPOKLM(ETH3D, TH3D, Z3D, U3D, CFILE, SS3D, RTR,   PP, IRC )
      IF( IRC .NE. 0 ) IRC = -8
C     per electron
      SMIS=(SS3S+SS3P+SS3D)/18.
      IF( SMIS .EQ. 0. ) SMIS = 1.E10
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
      SUBROUTINE INTPOKLM(ETHH, THE, Z, U, CFILE, SI, RTR, PP, IRC )
C
C     interpolates KLM cross sections with F functions
C
      REAL*4 F(4,40), ETHH, THE, Z, U, SI, RTR, PP, TH1, TH2, ETH1, ETH2
      REAL*4 F1, F2, F3, FTH, BL, B0, B1, B2, CL, C1, C2, C3, D_ETA, DL
      REAL*4 SLO, HELP1(2), HELP2(2), ETA_T, F_MAX, F_SMALL, AL, EL
      REAL*4 F_SMALL_T, RINT1D, RINT1DL, FINTPOL, FL, G_ETA, G_ETA_T, RL
      INTEGER*4 ID, IRC
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
      REAL*4 ETATH(40), THETA(4)
      COMMON /ETATH/  ETATH, THETA
C **********************************************************************
      CHARACTER*3 CFILE
      CHARACTER*14 CFILE_ALL
C
      REAL*4 FTHE1(4), FTHE2(4), THE_TAB(22), C1_TAB(22), C2_TAB(22)
      REAL*4 C3_TAB(22), ATAB(3), BTAB(3), CTAB(3)
      DATA THE_TAB /
     #   1.0, .95, .90, .85, .80, .75, .70, .68, .66, .64, .62,
     #   .60, .58, .56, .54, .52, .50, .48, .46, .44, .42, .40 /
      DATA C1_TAB /
     # 0.2834, 0.3264, 0.3784, 0.4422, 0.5211, 0.6200, 0.7458, 0.8056,
     # 0.8721, 0.9462, 1.0290, 1.1218, 1.2262, 1.3442, 1.4779, 1.6303,
     # 1.8046, 2.0050, 2.2369, 2.5065, 2.8221, 3.1943 /
      DATA C2_TAB /
     # 1.6476, 1.6806, 1.6972, 1.6887, 1.6427, 1.5404, 1.3538, 1.2465,
     # 1.1154, 0.9560, 0.7630, 0.5299, 0.2488,-0.0899,-0.4981,-0.9905,
     #-1.5852,-2.3054,-3.1798,-4.2455,-5.5499,-7.1547 /
      DATA C3_TAB /
     # 8.0519, 7.8541, 7.6611, 7.4946, 7.3903, 7.4064, 7.6383, 7.8236,
     # 8.0813, 8.4285, 8.8861, 9.4803,10.2433,11.2154,12.4471,14.0021,
     #15.9609,18.4262,21.5302,25.4435,30.3892,36.6612 /
      DATA ATAB / 22.222, 16., 4.028/
      DATA BTAB / 6.333, 8.667, 11.469/
      DATA CTAB / 4.741, 12.642, 3.330/
C *********************************************************************
C
      IRC = -1
      IF( ETHH .EQ. 0. .OR. U .EQ. 1. ) THEN
        SI = 0.
        RTR = 0.
        PP = 0.
        GOTO 999
      ENDIF
C
      IF( CFILE .EQ. 'fk' ) GOTO 10
      IF( CFILE .EQ. 'fl1' .OR. CFILE .EQ. 'fl2' ) GOTO 40
      IF( CFILE .EQ. 'fm1' .OR. CFILE .EQ. 'fm2' .OR. CFILE .EQ. 'fm3' )
     #         GOTO 70
 
C     Next line starts K shell interpolation
10    CONTINUE
      IF( ETHH .LT. .01 ) THEN
        B0 = 2912.711
        B1 = 19.* THE - 43.636
        B2 = 102.857 * ( 2.091*THE**2 - 10.778*THE + 12.308 )
        FTH = B0 * ETHH**4 * ( 1. + B1*ETHH + B2*ETHH**2 )
        FTH = FTH / THE
C
      ELSE IF( ETHH .GE. 0.01 .AND. ETHH .LE. 86.6 ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 20 ID=1, 40
          READ(9,*) F(1,ID),F(2,ID),F(3,ID),F(4,ID)
          IF( ETATH(ID) .GT. ETHH .AND. ID .GE. 2 ) GOTO 21
20      CONTINUE
21      CONTINUE
        IF( THE .LT. THETA(2)) THEN
           TH1=THETA(1)
           TH2=THETA(2)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(1,ID-1)
           F2=F(1,ID)
           F3=F(2,ID-1)
        ELSE IF( THE .GE. THETA(2) .AND. THE .LE. THETA(3) ) THEN
           TH1=THETA(2)
           TH2=THETA(3)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(2,ID-1)
           F2=F(2,ID)
           F3=F(3,ID-1)
        ELSE
           TH1=THETA(3)
           TH2=THETA(4)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(3,ID-1)
           F2=F(3,ID)
           F3=F(4,ID-1)
        ENDIF
        FTH = FINTPOL(ETHH, THE, TH1, TH2, ETH1, ETH2, F1, F2, F3 )
C
      ELSE IF( ETHH .GT. 86.6 ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 30 ID=1, 40
          READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
30      CONTINUE
31      CONTINUE
        C1 = RINT1D(THE, THE_TAB, C1_TAB, 22)
        C2 = RINT1D(THE, THE_TAB, C2_TAB, 22)
        C3 = RINT1D(THE, THE_TAB, C3_TAB, 22)
        F_MAX = RINT1DL(THE, THETA, FTHE1, 4)
        ETA_T = 86.6 * THE**2
        F_SMALL_T = F_MAX * ETA_T / THE
        G_ETA_T = C2 / 4. / ETA_T + C3 / 32. / ETA_T**2
        D_ETA = F_SMALL_T - C1 * LOG(ETA_T) + G_ETA_T
        ETA = ETHH * THE**2
        G_ETA = C2 / 4. / ETA + C3 / 32. / ETA**2
        F_SMALL = C1 * LOG(ETA) + D_ETA - G_ETA
        FTH = F_SMALL / ETA
      ENDIF
      GOTO 900
C
C     Next line starts L shell interpolation
40    CONTINUE
      IF( ETHH .LT. .01 ) THEN
        CLOSE(9)
        CFILE_ALL = 'CST'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
        READ(9,*) FTHE2(4),FTHE2(3),FTHE2(2),FTHE2(1)
        HELP2(1) = RINT1DL(THE, THETA, FTHE1, 4)
        HELP2(2) = RINT1DL(THE, THETA, FTHE2, 4)
        HELP1(1) = ETATH(1)
        HELP1(2) = ETATH(2)
        FTH  = RINT1DL(ETHH, HELP1, HELP2, 2 )
        FTH = FTH / THE
C
      ELSE IF( ETHH .GE. 0.01 .AND. ETHH .LE. 86.6 ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 50 ID=1, 40
          READ(9,*) F(1,ID),F(2,ID),F(3,ID),F(4,ID)
          IF( ETATH(ID) .GT. ETHH .AND. ID .GE. 2 ) GOTO 51
50      CONTINUE
51      CONTINUE
        IF( THE .LT. THETA(2)) THEN
           TH1=THETA(1)
           TH2=THETA(2)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(1,ID-1)
           F2=F(1,ID)
           F3=F(2,ID-1)
        ELSE IF( THE .GE. THETA(2) .AND. THE .LE. THETA(3) ) THEN
           TH1=THETA(2)
           TH2=THETA(3)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(2,ID-1)
           F2=F(2,ID)
           F3=F(3,ID-1)
        ELSE
           TH1=THETA(3)
           TH2=THETA(4)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(3,ID-1)
           F2=F(3,ID)
           F3=F(4,ID-1)
        ENDIF
        FTH = FINTPOL(ETHH, THE, TH1, TH2, ETH1, ETH2, F1, F2, F3 )
C
      ELSE IF( ETHH .GT. 86.6 ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 60 ID=1, 38
          READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
60      CONTINUE
61      CONTINUE
        READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
        READ(9,*) FTHE2(4),FTHE2(3),FTHE2(2),FTHE2(1)
        HELP2(1) = RINT1DL(THE, THETA, FTHE1, 4)
        HELP2(2) = RINT1DL(THE, THETA, FTHE2, 4)
        HELP1(1) = ETATH(39)
        HELP1(2) = ETATH(40)
        FTH = RINT1DL(ETHH, HELP1, HELP2, 2)
        FTH = FTH / THE
      ENDIF
      GOTO 900
C
C     Next line starts M shell interpolation
70    CONTINUE
      IF( ETHH .LT. .04 ) THEN
        IF( CFILE .EQ. 'FM1' ) THEN
          RL = 0.
          CL = CTAB(1)
          BL = BTAB(1)
          AL = ATAB(1)
        ELSE IF( CFILE .EQ. 'FM2' ) THEN
          RL = 1.
          CL = CTAB(2)
          BL = BTAB(2)
          AL = ATAB(2)
        ELSE IF( CFILE .EQ. 'FM3' ) THEN
          RL = 2.
          CL = CTAB(3)
          BL = BTAB(3)
          AL = ATAB(3)
        ENDIF
C
        DL = 1024. * 387420489. * 2.**(2.*RL) * 3.**(4.*RL)
     #     / (5.+RL) / (9.+2.*RL) * CL
        EL = 324. * (5.+RL) / (6.+RL) * (9.+2.*RL) / (11.+2.*RL) * AL
        FL = 36.* (5.+RL) / (6.+RL) * (9.+2.*RL) / (10.+2.*RL) * BL
C
        FTH = DL * ETHH**(4.+RL) * (1.-(EL-FL*THE)*ETHH)
        FTH = FTH / THE
C
      ELSE IF( ETHH .GE. 0.04 .AND. ETHH .LE. 100. ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 80 ID=1, 40
          READ(9,*) F(4,ID),F(3,ID),F(2,ID),F(1,ID)
          IF( ETATH(ID) .GT. ETHH .AND. ID .GE. 2 ) GOTO 81
80      CONTINUE
81      CONTINUE
        IF( THE .LT. THETA(2)) THEN
           TH1=THETA(1)
           TH2=THETA(2)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(1,ID-1)
           F2=F(1,ID)
           F3=F(2,ID-1)
        ELSE IF( THE .GE. THETA(2) .AND. THE .LE. THETA(3) ) THEN
           TH1=THETA(2)
           TH2=THETA(3)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(2,ID-1)
           F2=F(2,ID)
           F3=F(3,ID-1)
        ELSE
           TH1=THETA(3)
           TH2=THETA(4)
           ETH1=ETATH(ID-1)
           ETH2=ETATH(ID)
           F1=F(3,ID-1)
           F2=F(3,ID)
           F3=F(4,ID-1)
        ENDIF
C
        FTH = FINTPOL(ETHH, THE, TH1, TH2, ETH1, ETH2, F1, F2, F3 )
C
      ELSE IF( ETHH .GT. 100. ) THEN
        CLOSE(9)
        CFILE_ALL = 'cst'//CFILE//'.data'
        OPEN(9,FILE=CFILE_ALL,FORM='FORMATTED',STATUS='OLD')
        REWIND(9)
        DO 90 ID=1, 38
          READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
90      CONTINUE
91      CONTINUE
        READ(9,*) FTHE1(4),FTHE1(3),FTHE1(2),FTHE1(1)
        READ(9,*) FTHE2(4),FTHE2(3),FTHE2(2),FTHE2(1)
        HELP2(1) = RINT1DL(THE, THETA, FTHE1, 4)
        HELP2(2) = RINT1DL(THE, THETA, FTHE2, 4)
        HELP1(1) = ETATH(39)
        HELP1(2) = ETATH(40)
        FTH = RINT1DL(ETHH, HELP1, HELP2, 2)
        FTH = FTH / THE
      ENDIF
C     longitudinal cross.
900   CONTINUE
      IF( FTH .LE. 0. ) FTH = 0.
      SLO=7.038E8*FTH/Z**4
C     transverse/longitudinal
      RTR=(2.*LOG(GA)-BETA**2)/LOG(1022000.*BETA**2/U)
      SI=SLO*(1.+RTR)
C
      IRC = 0
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      FUNCTION FINTPOL(ETHH, THE, TH1, TH2, ETH1, ETH2, F1, F2, F3 )
C
C     TWO-DIMENSIONAL INTERPOLATION OF F
C
      REAL*4 FINTPOL, ETHH, THE, TH1, TH2, ETH1, ETH2, F1, F2, F3
      REAL*4 LT1, LT2, LE1, LE2, R12, B, L11, L12, PP, A
C **********************************************************************
      IF( ETH1 .EQ. 0.  .OR. ETH2 .EQ. 0. ) THEN
        FINTPOL = 0.
        GOTO 999
      ENDIF
      LT1=LOG(TH1)
      LT2=LOG(TH2)
      LE1=LOG(ETH1)
      LE2=LOG(ETH2)
      R12=F1*ETH1/F2/ETH2
      B=(R12*LE2-LE1)/(1.-R12)-2.*LT1
      L11=LE1+2.*LT1+B
      L12=LE1+2.*LT2+B
      if(  F3*L11/F1/L12 .le. 0. )
     # WRITE(6,*) F3*L11/F1/L12, f3, l11, f1, l12, LT1, LT2, b
      PP=LOG(F3*L11/F1/L12)/(LT1-LT2)
      A=F1*ETH1*TH1**PP/L11
C     this is F(Benka,Johnson)/THETA
      FINTPOL=A*(LOG(ETHH*THE*THE)+B)/ETHH/THE**(PP+1.)
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE BIPOCO1S( X, CX, ETH, THE, Z, BP )
C
C     binding and polarization correction
C     1s binding correction for targets
C
      REAL*4 G, X, CX, FF, FNPOL, PI/3.1415/, PH, ETH, THE, Z, BP
C
C **********************************************************************
C
      G = ( 1. + 9. * X + 31. * X**2 + 98. * X**3 + 12. * X**4
     #  + 25. * X**5 + 4.2 * X**6 + .515 * X**7 ) / (1. + X)**9
      IF( CX .GT. .035 ) THEN
         FF = FNPOL(CX)
      ELSE
         FF=-3. * PI / 4. * (2.*LOG(CX)+1.)
      ENDIF
      PH=FF/X/X/THE/SQRT(ETH)
      PH = 0.
      BP=2.*(G-PH)/Z/THE
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE BIPOCO2S( X, CX, ETH, THE, Z, BP )
C
C     binding and polarization correction
C     2s binding correction for targets
C
      REAL*4 G, X, CX, FF, FNPOL, PI/3.1415/, PH, ETH, THE, Z, BP
C
C **********************************************************************
      G = ( 1. + 9. * X + 31. * X**2 + 49. * X**3 + 162. * X**4
     #  + 63. * X**5 + 18. * X**6 + 1.97 * X**7 ) / (1. + X)**9
      IF( CX .GT. .035 ) THEN
         FF = FNPOL(CX)
      ELSE
         FF=-3. * PI / 4. * (2.*LOG(CX)+1.)
      ENDIF
      PH=FF/X/X/THE/SQRT(ETH)
      PH = 0.
      BP=2.*(G-PH)/Z/THE
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE BIPOCO2P( X, CX, ETH, THE, Z, BP )
C
C     binding and polarization correction
C     2P binding correction for targets
C
      REAL*4 G, X, CX, FF, FNPOL, PI/3.1415/, PH, ETH, THE, Z, BP
C
C **********************************************************************
      G = ( 1. + 10. * X + 45. * X**2 + 102. * X**3 + 331. * X**4
     #  + 67. * X**5 + 58. * X**6 + 7.8 * X**7 + .888 * X**8 )
     #  / (1. + X)**9
      IF( CX .GT. .035 ) THEN
         FF = FNPOL(CX)
      ELSE
         FF=-3. * PI / 4. * (2.*LOG(CX)+1.)
      ENDIF
      PH=FF/X/X/THE/SQRT(ETH)
      PH = 0.
      BP=2.*(G-PH)/Z/THE
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE CAPCROSS(ZP, ZT, SC, IRC )
C
C     subroutines for capt. cross
C
C     Eikonal approximation for relativistic non-radiative capture
C     based on analytical expression of Eichler. For K+L+M target
C     electrons into bare projectile K,L,M shells.
C     NOTATION: SEIK(I) eikonal cross. from all target shells into
C     bare proj. I shell. SEIK(I,K) eikonal cross. from target I to
C     bare proj. K shell.
c
C **********************************************************************
      REAL*4 Z2PFAC(3,3), OBKFAC(3,3), EIKFAC(3,3), EIK(3,3), MAG(3,3)
      REAL*4 ORB(3,3), CEIK(3,3), SC(3), EXPEZ, EXZT, EZP, EZP2, PEZ, PM
      REAL*4 ZFAC, ZPS, ZTS, Z1, Z2, Z2Q, EF, EI, ZP, ZT
      INTEGER*4 IL, IM, JP, JT, IRC
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
      IF( ZT .LT. 10. ) THEN
        IL=INT(ZT-2.)
      ELSE
        IL=8
      ENDIF
      IF( ZT .LT. 28. ) THEN
        IM=INT(ZT-10.)
      ELSE
        IM=18
      ENDIF
      IF( ZT .LT. 19. ) IM=0
      DO 10 JP = 1, 3
        ZPS=ZP/FLOAT(JP)
        DO 20 JT=1, 3
          IF( JT .EQ. 1 ) ZTS=ZT-.3
          IF( JT .EQ. 2 ) ZTS=(ZT-3.)/2.
          IF( JT .EQ. 3 ) ZTS=(ZT-19.)/3.
          IF( JT .EQ. 1 .AND. JP .EQ. 1 ) THEN
            Z2PFAC(JP,JT) = 1.
          ELSE
            Z2PFAC(JP,JT)=1.16
          ENDIF
          IF( ZPS .LE. ZTS ) THEN
            Z1=ZPS
            Z2=ZTS
            Z2Q=ZTS*Z2PFAC(JP,JT)
          ENDIF
          IF( ZPS .GT. ZTS ) THEN
            Z1=ZTS
            Z2=ZPS
            Z2Q=ZPS*Z2PFAC(JP,JT)
          ENDIF
          EI = SQRT(1.-ALPH*ALPH*Z2*Z2)
          EF = SQRT(1.-ALPH*ALPH*Z1*Z1)
          PM = ETA*(EF/GA-EI)/ALPH/ALPH
          EZP=ETA*Z2Q
          EZP2=EZP*EZP
          PEZ=PI*EZP
          EXPEZ=EXP(PEZ)
          EXZT=-2.*EZP*ATAN(-PM/Z2)
          ZFAC=Z1*Z2/(Z2*Z2+PM*PM)
          OBKFAC(JP,JT)=2.8E+07*128.*PI*ETA*ETA*ZFAC**5/GAG/GA/5.
          EIKFAC(JP,JT)=2.*PEZ*EXP(EXZT)/(EXPEZ-1./EXPEZ)
          EIK(JP,JT)=1.+5.*EZP*PM/Z2/4.+5.*EZP2*PM*PM/Z2/Z2/12.+EZP2/6.
          MAG(JP,JT)=-D2+5.*D2*D2/16.+5.*D2*GAG*Z2Q/Z2/8.+D2*EZP2/4.
     #              +5.*D2*D2*EZP2/48.
          ORB(JP,JT)=5.*PI*D1*ALPH*(Z1+Z2)*(1.-D2/2.)/18.-5.*D1*ALPH*Z2
     #             *EZP*(1.-D2/2.)/8.-5.*PI*D1*GAG*ALPH*Z1*Z2Q/18./Z2
     #             +5.*PI*D1*GAG*GAG*ALPH*Z1*Z2Q*Z2Q/28./Z2/Z2
     #             -5.*PI*D1*GAG*ALPH*(Z1+Z2-D2*Z1)*Z2Q/28./Z2
          CEIK(JP,JT)=OBKFAC(JP,JT)*EIKFAC(JP,JT)*(EIK(JP,JT)+
     #                MAG(JP,JT)+ORB(JP,JT))
20      CONTINUE
        SC(JP)=JP*JP*(2.*CEIK(JP,1)+IL*CEIK(JP,2)+IM*CEIK(JP,3))
10    CONTINUE
      IRC = 0
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE RECCORK(Z, U, SRKBEST, IRC )
C
C     REC subroutine for K shell
C
      INTEGER*4 IRC
      REAL*4 Z, U, B, NU, E, KH, NS, NB, DB, RECF, AUNP, SRK, AUSA
      REAL*4 SRKSA, SRKBEST
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
C     start of K REC calculation
      B=U/MC2
      NU=B+GA-1.
      E=B/NU
      KH=SQRT(B/(GA-1.))
      NS=1.
      NB=EXP(-4.*NS*KH*ATAN(1./KH))
      DB=1.-EXP(-2.*PI*NS*KH)
      RECF=9614.*E*E*B/(GA-1.)
      AUNP=2.*KH/(1.-KH*KH-NU*NU)
C     SRK for dipole only (Bethe-Salpeter expression)
      SRK=RECF*NB/DB
      AUSA=1.+3./4.*GA*(GA-2.)/(GA+1.)
     #     *(1.-LOG((1.+BETA)/(1.-BETA))/2./BETA/GA/GA)
C     Sauter cross. per electron
      SRKSA=3.77E-09*Z**5*BETA*GA/NU/NU/NU*AUSA/2.
      IF( SRK .GT. SRKSA ) THEN
        SRKBEST=(SRK+SRKSA)/2.
      ELSE
        SRKBEST=SRKSA
      ENDIF
      IRC = 0
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE RECCORLM(U, SRL, SRM, IRC )
C
C     REC subroutine for LM shell
C
      INTEGER*4 IRC
      REAL*4   U, B, NU, E, LH, NS, NB, DB, RECF, SRL, SRM
C **********************************************************************
      REAL*4         PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
      COMMON /CONST/ PI, U0, MC2, S0, ALPH, GA, BSQ, BETA, BSA,
     #               D1, D2, GAG, ETA
C **********************************************************************
C
C     start of LM REC calculation
      B=U/MC2
      NU=B+GA-1.
      E=B/NU
      LH=SQRT(B/(GA-1.))
      NS=2.
      NB=EXP(-4.*NS*LH*ATAN(1./LH))
      DB=1.-EXP(-2.*PI*NS*LH)
      RECF=9614.*E*E*B/(GA-1.)
C     dipole only
      SRL=RECF*NB/DB*8.*(1.+6.*E+8.*E*E)
      SRM=8./27.*SRL
      IRC = 0
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE BBRANGE(A, Z, AT, ZT, T, RG)
C
C     CALCULATES RANGES WITH E. HANELT FIT (E. HANELT, PHD THESIS)
C
      REAL*4 Y1,Y2,Y3,Y4,Y5,F,A,Z,AT,ZT,T,RG
      REAL*4 COEFF,COEFFI,FCOR
      DIMENSION F(10)
C
C *********************************************************************
      IF( T .LE. 1.0E-6 ) THEN
        RG = 0.
        GOTO 999
      ENDIF
C
      CALL FDATA( ZT, F, COEFFI, COEFF)
C
      Y1 = 1.0+F(1)*Z+F(2)*Z**2+F(3)*Z**3+F(4)*Z**4
      Y2 = F(5)+F(6)*Z
      Y3 = F(7)+F(8)*Z
      Y4 = F(9)+F(10)*Z
      Y5 = Y3/(2.0*Y4)
C
C     MATTER COEFFICIENT
      COEFF = (AT*COEFFI)/ZT**COEFF
C
C     RANGE
      CALL FCORR(Z, FCOR)
      RG =(A/Z**2)*10.0**(Y1*(Y2+Y3*LOG10(T)+Y4*LOG10(T)**2))*COEFF
     #   * FCOR
      IF( RG .LT. 0. ) RG = 0.
C
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE ENERGY(A, Z, AT, ZT, RG, T)
C
C     CALCULATES ENERGIES FROM RANGES WITH EH FIT (E. HANELT, PHD THESIS
C
      REAL*4 Y1,Y2,Y3,Y4,Y5,F,A,Z,AT,ZT,T,RG
      REAL*4 COEFFI,COEFF,FCOR
      DIMENSION F(10)
C
C *********************************************************************
      IF( RG .LE. 1.0E-9 ) THEN
        T = 0.
        GOTO 999
      ENDIF
C
      CALL FDATA( ZT, F, COEFFI, COEFF)
C
      Y1 = 1.0+F(1)*Z+F(2)*Z**2+F(3)*Z**3+F(4)*Z**4
      Y2 = F(5)+F(6)*Z
      Y3 = F(7)+F(8)*Z
      Y4 = F(9)+F(10)*Z
      Y5 = Y3/(2.0*Y4)
C
C     MATTER COEFFICIENT
      COEFF = (AT*COEFFI)/ZT**COEFF
C     ENERGY
      CALL FCORR(Z, FCOR)
      T = 10.0**(-Y5-SQRT(Y5**2-Y2/Y4+LOG10(RG/FCOR/
     #    COEFF*Z**2/A)/(Y1*Y4)))
      IF( T .LT. 0. ) T = 0.
C
 999  RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE FDATA(Z,F,COEFFI,COEFF)
C
C     PARAMETERS FOR THE RANGE-ENERGY FITS
C
      REAL*4 Z,F,COEFFI, COEFF
      DIMENSION F(1:10)
C
C *********************************************************************
      IF (Z.LE.5.0) THEN
        F(1) = -0.128482E-03
        F(2) = -0.173612E-05
        F(3) = 0.889892E-07
        F(4) = -0.705115E-09
        F(5) = -0.553492E+00
        F(6) = 0.912049E-02
        F(7) = 0.268184E+01
        F(8) = -0.529303E-02
        F(9) = -0.210108E+00
        F(10) = 0.774360E-03
        COEFFI = 4.0**0.98/9.012
        COEFF  = 0.98
      END IF
      IF ((Z.GT.5.0).AND.(Z.LE.9.0)) THEN
        F(1) = 0.667801E-03
        F(2) = -0.392137E-05
        F(3) = 0.136917E-06
        F(4) = -0.972996E-09
        F(5) = -0.490202E+00
        F(6) = 0.751599E-02
        F(7) = 0.261390E+01
        F(8) = -0.600822E-02
        F(9) = -0.199549E+00
        F(10) = 0.731880E-03
        COEFFI = 6.0**0.98/12.011
        COEFF  = 0.98
      END IF
      IF ((Z.GT.9.0).AND.(Z.LE.32.0)) THEN
        F(1) = -0.668659E-04
        F(2) = -0.185311E-05
        F(3) = 0.873192E-07
        F(4) = -0.690141E-09
        F(5) = -0.530758E+00
        F(6) = 0.898953E-02
        F(7) = 0.268916E+01
        F(8) = -0.533772E-02
        F(9) = -0.214131E+00
        F(10) = 0.773008E-03
        COEFFI = 13.0**0.90/26.982
        COEFF  = 0.90
      END IF
      IF ((Z.GT.32.).AND.(Z.LE.64.0)) THEN
        F(1) =  1.23639E-03
        F(2) = -6.13893E-06
        F(3) = 1.84116E-07
        F(4) = -1.20551E-09
        F(5) = -0.263421E+00
        F(6) = 6.34349E-03
        F(7) = 2.61081E+00
        F(8) = -6.38315E-03
        F(9) = -0.204813E+00
        F(10) = 6.63267E-04
        COEFFI = 50.0**0.88/118.69
        COEFF  = 0.88
      END IF
      IF ((Z.GT.64.0).AND.(Z.LE.76.0)) THEN
        F(1) = 0.199249E-04
        F(2) = -0.227944E-05
        F(3) = 0.105063E-06
        F(4) = -0.829122E-09
        F(5) = -0.325062E+00
        F(6) = 0.975017E-02
        F(7) = 0.268814E+01
        F(8) = -0.607419E-02
        F(9) = -0.218986E+00
        F(10) = 0.869283E-03
        COEFFI = 73.0**0.88/180.95
        COEFF  = 0.88
      END IF
      IF (Z.GT.76.0) THEN
        F(1) = -0.375861E-03
        F(2) = -0.373902E-05
        F(3) = 0.148861E-06
        F(4) = -0.112159E-08
        F(5) = -0.166220E+00
        F(6) = 0.126920E-01
        F(7) = 0.259061E+01
        F(8) = -0.725322E-02
        F(9) = -0.202004E+00
        F(10) = 0.117942E-02
        COEFFI = 82.0**0.80/207.2
        COEFF  = 0.80
      END IF
C
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE FCORR(Z,F)
C
C     CALCULATES A CORRECTION TO THE RANGE FIT OF ECKHARD HANELT
C     based on measured dE/dx values from thesis Christoph Scheidenberg
C     Karl-Heinz Schmidt, 19. 12. 1995
C
      REAL*4 Z,R,F
C
C *********************************************************************
      R = Z**2 / 1000.
      F = 1./ ( 0.965735686 + 9.79114E-3 * R + 3.17099E-3 * R**2
     #       - 6.71227E-4 * R**3 + 2.28409E-5 * R**4 )
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE ELEMENT( Z, A, CZ, IOPT, IRC )
C
C     RETURNS ELEMENT SYMBOL FROM NUCLEAR CHARGE
C
C     IOPT 1: CHARGE NUMBER TO ELEMENT SYMBOL CONVERSION
C     IOPT 2: ELEMENT SYMBOL TO CHARGE NUMBER CONVERSION
C     IOPT 3: CHARGE NUMBER TO ELEMENT SYMBOL CONVERSION
C     IOPT 4: ELEMENT SYMBOL TO CHARGE NUMBER CONVERSION
C     IOPT 3 ET 4: INTEGER CUTTING FOR BEAMS
C     IOPT 5: RETURNS Z AND CZ FOR A GIVEN MASS NUMBER A
C     IOPT 6: Z TO SYMBOL CONVERSION
C
      REAL*4 Z, A, MASS(97)
      INTEGER*4 IOPT, IRC, I
      CHARACTER*2 CZ, CZDATA(97)
C
C **********************************************************************
      DATA CZDATA/
     # 'H ', 'HE',
     # 'LI', 'BE', 'B ', 'C ', 'N ', 'O ', 'F ', 'NE',
     # 'NA', 'MG', 'AL', 'SI', 'P ', 'S ', 'CL', 'AR',
     # 'K ', 'CA', 'SC', 'TI', 'V ', 'CR', 'MN', 'FE', 'CO', 'NI',
     # 'CU', 'ZN', 'GA', 'GE', 'AS', 'SE', 'BR', 'KR',
     # 'RB', 'SR', 'Y ', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD',
     # 'AG', 'CD', 'IN', 'SN', 'SB', 'TE', 'J ', 'XE',
     # 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB',
     # 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W ', 'RE', 'OS',
     # 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'PO', 'AT', 'RN',
     # 'FR', 'RA', 'AC', 'TH', 'PA', 'U ', 'NP', 'PU', 'AM', 'CM', 'MY'/
C
      DATA MASS/
     #  1.0,    4.0,
     #  6.94,   9.01, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
     #  23.00, 24.31, 26.98, 28.09, 30.97, 32.06, 35.45, 39.95,
     #  39.10, 40.08, 44.96, 47.88, 50.94, 52.00, 54.94, 55.84, 58.93,
     #  58.69,
     #  63.55, 65.40, 69.72, 72.59, 74.92, 78.96, 79.90, 83.80,
     #  85.47, 87.62, 88.91, 91.22, 92.91, 95.93, 98.  , 101.07, 102.91,
     # 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90,
     # 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 147.,
     # 150.36, 151.97, 157.25, 158.92, 162.50, 164.93, 167.26, 168.93,
     # 173.03, 174.97, 178.49, 180.95, 183.85, 186.21, 190.20, 192.22,
     # 195.08, 196.97, 200.60, 204.38, 207.20, 208.98, 203., 209., 211.,
     # 212., 213., 222., 232., 231., 238., 237., 244., 243., 247.,
     # 13.72/
C
C **********************************************************************
C
      CALL UPPERCASE(CZ)
C
      IF( IOPT .EQ. 1 .OR. IOPT .EQ. 3 ) THEN
        IF( Z .EQ. 6.6 ) THEN
          CZ = CZDATA(97)
          GOTO 990
        ENDIF
        DO I = 1, 96
          IF( INT(Z) .EQ. I ) THEN
            CZ = CZDATA(I)
            IRC = 0
            GOTO 990
          ENDIF
        ENDDO
C
      ELSE IF( IOPT .EQ. 2 .OR. IOPT .EQ. 4 ) THEN
        DO I = 1, 97
          IF( CZ .EQ. CZDATA(I) ) THEN
            Z = FLOAT(I)
            IF( CZ .EQ. 'MY' ) Z = 6.6
            IRC = 0
            GOTO 990
          ENDIF
        ENDDO
      ELSE IF( IOPT .EQ. 5 ) THEN 
        DO I = 1, 97
          IF( A .LE. MASS(I) ) THEN
            Z = FLOAT(I) 
            CZ = CZDATA(I)
            GOTO 999
          ENDIF
        ENDDO 
      ELSE IF( IOPT .EQ. 6 ) THEN 
        CZ = CZDATA(NINT(Z))
        GOTO 999
      ENDIF
C
      WRITE(6,*) ' <E>: Element not forseen '
      IRC = -1
990   CONTINUE
      A = MASS( NINT(Z) )
      IF( Z .EQ. 6.6 ) A = MASS(97)
      IF( IOPT .EQ. 3 .OR. IOPT .EQ. 4 ) A = NINT(A)
C
999   RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE UPPERCASE(CARG)
C
C     transform lowercase characters to uppercase characters
C
      CHARACTER*(*) CARG
      INTEGER*4 ILEN, I
C
C *********************************************************************
      ILEN = LEN(CARG)
      DO I = 1, ILEN
        IF( CARG(I:I) .EQ. 'a' ) CARG(I:I) = 'A'
        IF( CARG(I:I) .EQ. 'b' ) CARG(I:I) = 'B'
        IF( CARG(I:I) .EQ. 'c' ) CARG(I:I) = 'C'
        IF( CARG(I:I) .EQ. 'd' ) CARG(I:I) = 'D'
        IF( CARG(I:I) .EQ. 'e' ) CARG(I:I) = 'E'
        IF( CARG(I:I) .EQ. 'f' ) CARG(I:I) = 'F'
        IF( CARG(I:I) .EQ. 'g' ) CARG(I:I) = 'G'
        IF( CARG(I:I) .EQ. 'h' ) CARG(I:I) = 'H'
        IF( CARG(I:I) .EQ. 'i' ) CARG(I:I) = 'I'
        IF( CARG(I:I) .EQ. 'j' ) CARG(I:I) = 'J'
        IF( CARG(I:I) .EQ. 'k' ) CARG(I:I) = 'K'
        IF( CARG(I:I) .EQ. 'l' ) CARG(I:I) = 'L'
        IF( CARG(I:I) .EQ. 'm' ) CARG(I:I) = 'M'
        IF( CARG(I:I) .EQ. 'n' ) CARG(I:I) = 'N'
        IF( CARG(I:I) .EQ. 'o' ) CARG(I:I) = 'O'
        IF( CARG(I:I) .EQ. 'p' ) CARG(I:I) = 'P'
        IF( CARG(I:I) .EQ. 'q' ) CARG(I:I) = 'Q'
        IF( CARG(I:I) .EQ. 'r' ) CARG(I:I) = 'R'
        IF( CARG(I:I) .EQ. 's' ) CARG(I:I) = 'S'
        IF( CARG(I:I) .EQ. 't' ) CARG(I:I) = 'T'
        IF( CARG(I:I) .EQ. 'u' ) CARG(I:I) = 'U'
        IF( CARG(I:I) .EQ. 'v' ) CARG(I:I) = 'V'
        IF( CARG(I:I) .EQ. 'w' ) CARG(I:I) = 'W'
        IF( CARG(I:I) .EQ. 'x' ) CARG(I:I) = 'X'
        IF( CARG(I:I) .EQ. 'y' ) CARG(I:I) = 'Y'
        IF( CARG(I:I) .EQ. 'z' ) CARG(I:I) = 'Z'
      ENDDO
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      FUNCTION RINT1D( X, XTAB, YTAB, N )
C
C     INTERPOLATION FOR LINEAR FUNCTIONS
C
      INTEGER*4 N, I, ISAVE
      REAL*4 X, XTAB(N), YTAB(N), RINT1D, Y, XX
C
C *********************************************************************
      DO 10 I = 1, N
        IF( X .EQ. XTAB(I) ) THEN
           Y = YTAB(I)
           GOTO 9999
        ENDIF
10    CONTINUE
C
      IF( X .LT. XTAB(1) ) THEN
         ISAVE = 1
         GOTO 100
      ENDIF
C
      IF( X .GT. XTAB(N) ) THEN
         ISAVE = N - 1
         GOTO 100
      ENDIF
C
      DO 11 I = 1, N
        IF( X .LT. XTAB(I) ) THEN
           ISAVE = I - 1
           GOTO 100
        ENDIF
11    CONTINUE
C
100   CONTINUE
C
      XX = X - XTAB(ISAVE)
      Y = YTAB(ISAVE) + XX * (YTAB(ISAVE+1) - YTAB(ISAVE)) /
     #                       (XTAB(ISAVE+1) - XTAB(ISAVE))
9999  RINT1D = Y
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      FUNCTION RINT1DL( X, XTAB, YTAB, N )
C
C     INTERPOLATION FOR LOGARITHMIC FUNCTIONS
C
      INTEGER*4 N, I, ISAVE
      REAL*4 X, XTAB(N), YTAB(N), RINT1DL, Y, XX
C
C *********************************************************************
      IF( X .EQ. 0. ) X = 1.E-25
      IF( X .LT. 0. ) THEN
        WRITE(6,*) '<E> RINT1DL: NEGATIVE VALUES NOT ALLOWED!'
        RINT1DL = 0.
        GOTO 9999
      ENDIF
C
      DO 10 I = 1, N
        IF( XTAB(I) .LT. 0. .OR. YTAB(I) .LT. 0. ) THEN
          WRITE(6,*) '<E> RINT1DL: NEGATIVE VALUES NOT ALLOWED!'
          RINT1DL = 0.
          GOTO 9999
        ENDIF
        IF( XTAB(I) .EQ. 0. ) XTAB(I) = 1.E-25
        IF( YTAB(I) .EQ. 0. ) YTAB(I) = 1.E-25
        IF( X .EQ. XTAB(I) ) THEN
           Y = LOG(YTAB(I))
           GOTO 9990
        ENDIF
10    CONTINUE
C
      IF( X .LT. XTAB(1) ) THEN
         ISAVE = 1
         GOTO 100
      ENDIF
C
      IF( X .GT. XTAB(N) ) THEN
         ISAVE = N - 1
         GOTO 100
      ENDIF
C
      DO 11 I = 1, N
        IF( X .LT. XTAB(I) ) THEN
           ISAVE = I - 1
           GOTO 100
        ENDIF
11    CONTINUE
C
100   CONTINUE
C
      XX = LOG(X) - LOG(XTAB(ISAVE))
      Y = LOG(YTAB(ISAVE)) + XX
     #  * (LOG(YTAB(ISAVE+1)) - LOG(YTAB(ISAVE))) /
     #                       (LOG(XTAB(ISAVE+1)) - LOG(XTAB(ISAVE)))
9990  RINT1DL = EXP(Y)
9999  RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PCHA( CSTR, CTR )
C
C     PROMPTING - SUBROUTINE FOR CHARACTERS
C
      CHARACTER*(*)  CSTR, CTR
      CHARACTER*10   CHELP
C
C *********************************************************************
      WRITE(6, 1) CSTR , CTR
1     FORMAT(' ',A,': ',A)
10    READ( 5, FMT='(A)', ERR=11, END=11 ) CHELP
      IF( CHELP .EQ. ' ' ) RETURN
C
      CALL NOLBL( CHELP )
      IF( CHELP .NE. ' ' ) CTR = CHELP
C
11    CONTINUE
 
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PILO( C_STR, I_INT  )
C
C     PROMPTING - SUBROUTINE FOR INTEGER 
C
      CHARACTER*(*)  C_STR
      CHARACTER*(80) C_REAL
      INTEGER*4 I_INT
C
C *********************************************************************
100   WRITE(6, 10) C_STR , I_INT
10    FORMAT(' ',A,' ',I5)
      READ( 5, FMT='(A)', ERR=11, END=11) C_REAL
      IF( C_REAL .EQ. ' ' ) RETURN
C
      IF( C_REAL .NE. ' ' ) THEN
        IF( C_REAL(1:1) .EQ. ' ' ) CALL NOLBL(C_REAL)
      ENDIF
C
      IF( C_REAL(1:1) .NE. '+' .AND. C_REAL(1:1) .NE. '-' .AND.
     #    C_REAL(1:1) .NE. '1' .AND. C_REAL(1:1) .NE. '2' .AND.
     #    C_REAL(1:1) .NE. '3' .AND. C_REAL(1:1) .NE. '4' .AND.
     #    C_REAL(1:1) .NE. '5' .AND. C_REAL(1:1) .NE. '6' .AND.
     #    C_REAL(1:1) .NE. '7' .AND. C_REAL(1:1) .NE. '8' .AND.
     #    C_REAL(1:1) .NE. '9' .AND. C_REAL(1:1) .NE. '0') THEN
         WRITE(6,*) '<E>: Integer number required!!!!!!'
         WRITE(6,*)
         GOTO 100
      ENDIF
C
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_CHIN( C_REAL, I_INT )
        RETURN
      ENDIF
C
11    CONTINUE
 
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PILO10( C_STR, IST, IMAX)
C
C     PROMPTING - SUBROUTINE FOR INTEGER 
C
      CHARACTER*(*)  C_STR
      CHARACTER*(80) C_REAL, C_SUB(30)
      INTEGER*4 IST(10), NDUM(30), NSUB, NMIN, NMAX, K, IMAX, I
C
C *********************************************************************
      DO I = 1, 10
        IF( IST(I) .GE. 0 ) IMAX = I
      ENDDO
C
      WRITE(6, 10) C_STR , (IST(I), I=1, IMAX)
10    FORMAT(' ',A,' ',I4, I4, I4, I4, I4, I4, I4, I4, I4, I4 )
      READ( 5, FMT='(A)', ERR=11, END=11) C_REAL
      IF( C_REAL .EQ. ' ' ) RETURN
C
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_SSTR( C_REAL, C_SUB, NDUM, NSUB )
C
        IF( C_SUB(2) .EQ. '-' ) THEN
           CALL C_CHIN( C_SUB(1), IST(1) )
           NMIN = MIN(IST(1), 28)
           NMAX = MIN(IST(1)+IMAX-1, 28)
           K = 0
           DO I  = NMIN, NMAX
              K = K + 1
              IF( K .GT. 10 ) GOTO 11
              IST(K) = I
           ENDDO
           IMAX = K
           IF( K .GE. 10 ) GOTO 11
           DO I = K+1, 10
             IST(I) = -1
           ENDDO
           GOTO 11
        ENDIF
C
        CALL C_CHIN( C_SUB(1), IST(1) )
        IF(C_SUB(2) .NE. ' ' ) CALL C_CHIN( C_SUB(2), IST(2) )
        IF(C_SUB(3) .NE. ' ' ) CALL C_CHIN( C_SUB(3), IST(3) )
        IF(C_SUB(4) .NE. ' ' ) CALL C_CHIN( C_SUB(4), IST(4) )
        IF(C_SUB(5) .NE. ' ' ) CALL C_CHIN( C_SUB(5), IST(5) )
        IF(C_SUB(6) .NE. ' ' ) CALL C_CHIN( C_SUB(6), IST(6) )
        IF(C_SUB(7) .NE. ' ' ) CALL C_CHIN( C_SUB(7), IST(7) )
        IF(C_SUB(8) .NE. ' ' ) CALL C_CHIN( C_SUB(8), IST(8) )
        IF(C_SUB(9) .NE. ' ' ) CALL C_CHIN( C_SUB(9), IST(9) )
        IF(C_SUB(10).NE. ' ' ) CALL C_CHIN( C_SUB(10),IST(10))
        RETURN
      ENDIF
C
11    CONTINUE
 
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PRSO( C_STR, R_REAL, I)
C
C     PROMPTING - SUBROUTINE 
C
      INTEGER   I
      CHARACTER*(*)  C_STR
      CHARACTER*(80) C_REAL
      REAL*4   R_REAL
C
C *********************************************************************
100   IF( I .LT. 0 ) THEN
          WRITE(6,*) ' <E>: I < 0 IN SUBROUTINE PRSO'
          RETURN
      ENDIF
      IF( I .GT. 6 ) I = 6
C
      IF( I .EQ. 0 ) THEN
        WRITE(6, 1) C_STR , R_REAL
1       FORMAT(' ',A,' ',G12.0)
        GOTO 10
      ENDIF
      IF( I .EQ. 1 ) THEN
        WRITE(6, 2) C_STR , R_REAL
2       FORMAT(' ',A,' ',G12.1)
        GOTO 10
      ENDIF
      IF( I .EQ. 2 ) THEN
        WRITE(6, 3) C_STR , R_REAL
3       FORMAT(' ',A,' ',G12.2)
        GOTO 10
      ENDIF
      IF( I .EQ. 3 ) THEN
        WRITE(6, 4) C_STR , R_REAL
4       FORMAT(' ',A,' ',G12.3)
        GOTO 10
      ENDIF
      IF( I .EQ. 4 ) THEN
        WRITE(6, 5) C_STR , R_REAL
5       FORMAT(' ',A,' ',G12.4)
        GOTO 10
      ENDIF
      IF( I .EQ. 5 ) THEN
        WRITE(6, 6) C_STR , R_REAL
6       FORMAT(' ',A,' ',G12.5)
        GOTO 10
      ENDIF
      IF( I .EQ. 6 ) THEN
        WRITE(6, 7) C_STR , R_REAL
7       FORMAT(' ',A,' ',G12.6)
        GOTO 10
      ENDIF
10    READ( 5, FMT='(A)', ERR=11, END=11 ) C_REAL
      IF( C_REAL .EQ. ' ' ) RETURN
C
      IF( C_REAL .NE. ' ' ) THEN
        IF( C_REAL(1:1) .EQ. ' ' ) CALL NOLBL(C_REAL)
      ENDIF
C
      IF( C_REAL(1:1) .NE. '-' .AND. C_REAL(1:1) .NE. '+' .AND.
     #    C_REAL(1:1) .NE. '1' .AND. C_REAL(1:1) .NE. '2' .AND.
     #    C_REAL(1:1) .NE. '3' .AND. C_REAL(1:1) .NE. '4' .AND.
     #    C_REAL(1:1) .NE. '5' .AND. C_REAL(1:1) .NE. '6' .AND.
     #    C_REAL(1:1) .NE. '7' .AND. C_REAL(1:1) .NE. '8' .AND.
     #    C_REAL(1:1) .NE. '9' .AND. C_REAL(1:1) .NE. '0' .AND.
     #    C_REAL(1:1) .NE. '.') THEN
         WRITE(6,*) '<E>: Real number required!!!!!!!'
         WRITE(6,*)
         GOTO 100
      ENDIF
C 
      IF( C_REAL .NE. ' ' ) THEN
        CALL C_CHRE( C_REAL, R_REAL )
        RETURN
      ENDIF
C
11    CONTINUE
 
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PRTC( C_STR, I_INT  )
C
C     PROMPTING - SUBROUTINE FOR TEXT STRING
C
      CHARACTER*(*)  C_STR
      INTEGER*4 I_INT
C
C *********************************************************************
      WRITE(I_INT, 10) C_STR
10    FORMAT(' ',A,' ' )
C
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE PYES( C_STR, C_ANS  )
C
C     PROMPTING - SUBROUTINE FOR QUESTIONS
C
      CHARACTER*(*)  C_STR
      CHARACTER*(80) C_REAL
      CHARACTER*(1) C_ANS
C
C *********************************************************************
1     WRITE(6, 10) C_STR, C_ANS
10    FORMAT(' ',A,'?   --> ',A,'?')
      READ( 5, FMT='(A)',ERR=100, END=100 ) C_REAL
C
      IF( C_REAL .EQ. ' ' ) C_REAL = C_ANS
C
      IF( C_REAL .NE. ' ' ) THEN
          CALL NOLBL(C_REAL)
          C_ANS = C_REAL(1:1)
          IF(  C_ANS .EQ. 'J' .OR. C_ANS .EQ. 'Y' .OR. C_ANS .EQ. 'O'
     #    .OR. C_ANS .EQ. 'j' .OR. C_ANS .EQ. 'y' .OR. C_ANS .EQ. 'o')
     #      THEN
            C_ANS = 'Y'
            GOTO 100
          ENDIF
          IF( C_ANS .EQ. 'N' .OR. C_ANS .EQ. 'n') THEN
            GOTO 100
          ENDIF
          WRITE(6,*) '<E>: Unexpected answer! Answer "y" or "n"'
          GOTO 1
      ENDIF
C
100   CONTINUE
 
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE NOLBL(CSTR)
C
C     SUPPRESSES LEADING BLANKS OF STRING CSTR
C
      CHARACTER*(*) CSTR
      CHARACTER*1   CWORK
      INTEGER*4     ILEN, I
C
C *********************************************************************
      ILEN = LEN(CSTR)
      DO 1 I = 1,ILEN
         CWORK = CSTR(I:I)
         IF( CWORK .NE. ' ' ) THEN
             CSTR = CSTR(I:)
             GOTO 2
         ENDIF
    1 CONTINUE
    2 CONTINUE
C
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE C_SSTR(C_KEEP, C_SUB, N_LBLK, N_SUB )
C
C     DECOMPOSES STRING C_LINE INTO <= 30 SUBSTRINGS C_SUB(XX)
C     MAXIMUM  TOTAL CHARACTER LENGTH = 75
C     N_SUB = ACTUAL NUMBER OF SUBSTRINGS
C     N_LBLK = NUMBER OF LEADING BLANKS FOR EACH SUBSTRING
C
      CHARACTER*(*) C_KEEP
      CHARACTER*250 C_LINE
      CHARACTER*(*) C_SUB(30)
      CHARACTER*250 C_WORD
      CHARACTER*1   C_L
      INTEGER*4     N_SUB, J, I, N_LBLK(30)
C
C *********************************************************************
      C_LINE = C_KEEP
      N_SUB = 0
C
      DO 5  I = 1, 30
         C_SUB(I) = ' '
   5  CONTINUE
C
      DO 10 J = 1, 30
         C_WORD = ' '
C
C  **    NUMBER OF BLANKS
C
         DO 11 I = 1, 250
            C_L = C_LINE(I:I)
            IF( C_L .NE. ' ' .AND. C_L .NE. ',' ) THEN
               N_LBLK(J) = I - 1
               C_LINE = C_LINE(I:)
               GOTO 12
            ENDIF
   11    CONTINUE
         GOTO 20
 
C  **    DECOMPOSITION IN SUBSTRINGS
C
   12    DO 15 I = 1, 250
            C_L = C_LINE(I:I)
            IF( C_L .NE. ' ' ) C_WORD(I:I) = C_L
            IF( C_L .EQ. ' ' .OR. C_L .EQ. ',' ) THEN
               C_LINE = C_LINE(I:)
               GOTO 16
            ENDIF
  15     CONTINUE
  16     C_SUB(J) = C_WORD
         N_SUB = N_SUB + 1
  10  CONTINUE
  20  RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE C_INCH(  I_INT , C_STR )
C
C     CONVERSION: INTEGER VARIABLE  -> CHARACTER - VARIABLE
C
      CHARACTER*(*)  C_STR
      CHARACTER*2    C_FORM
      CHARACTER*1    C_WORK, C_I
      INTEGER*4      I_INT, J
C
C *********************************************************************
      WRITE(C_STR ,FMT='(I8)'  ) I_INT
C
C     CUTS OFF UNNECESSARY SPACES
C
      DO 1 J = 1,8
        C_WORK = C_STR(J:J)
        IF( C_WORK .NE. ' ') THEN
          WRITE( C_I, FMT=('(I1)' ) ) 9 - J
          C_FORM = 'I'//C_I
          WRITE( C_STR, FMT='('//C_FORM//')' )  I_INT
        ENDIF
    1 CONTINUE
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE C_RECH(  R_REAL, C_STR )
C
C     CONVERSION: REAL VARIABLE  -> CHARACTER - VARIABLE
C
      CHARACTER*8    C_DUMMY
      CHARACTER*(*)  C_STR
      REAL*4         R_REAL
C
C *********************************************************************
      C_DUMMY = C_STR
      WRITE( C_DUMMY, FMT='(F8.4)' ) R_REAL
      CALL NOLBL(C_DUMMY)
      C_STR = C_DUMMY
C
      RETURN
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE C_CHIN( C_STR, I_INT  )
C
C     CONVERSION: CHARACTER VARIABLE  -> INTEGER - VARIABLE
C
      CHARACTER*(*)  C_STR
      INTEGER*4      I_INT
      REAL*4         R_INT
C
C *********************************************************************
      CALL C_CHRE( C_STR, R_INT)
      I_INT = NINT(R_INT)
C
      END
C
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C MFA the following subroutine is changed for MF version
C
      SUBROUTINE C_CHRE( C_STR, R_REAL )
C
C     CONVERSION: CHARACTER VARIABLE  -> REAL - VARIABLE
C
      CHARACTER*(*)  C_STR
      REAL*4         R_REAL
C
C *********************************************************************
      READ( C_STR , * ) R_REAL
C
      END
C
C $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
