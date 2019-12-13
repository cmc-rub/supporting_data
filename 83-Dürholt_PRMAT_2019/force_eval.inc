&FORCE_EVAL
  METHOD  QS
  STRESS_TENSOR  ANALYTICAL
  &DFT
    BASIS_SET_FILE_NAME /home/johannes/sandbox/cp2k-5.1/data/GTH_BASIS_SETS
    BASIS_SET_FILE_NAME /home/johannes/sandbox/cp2k-5.1/data/BASIS_MOLOPT
    POTENTIAL_FILE_NAME /home/johannes/sandbox/cp2k-5.1/data/POTENTIAL
    &SCF
      MAX_SCF  30
      EPS_SCF     1.0E-6
      SCF_GUESS  ATOMIC
      &MIXING  T
        METHOD  PULAY_MIXING
        ALPHA     5.0000000000000000E-01
        NBUFFER  5
      &END MIXING
      &PRINT
        &RESTART  OFF
        &END RESTART
      &END PRINT
      &OT ON
        PRECONDITIONER FULL_SINGLE_INVERSE
        MINIMIZER DIIS
      &END OT
      &OUTER_SCF
        MAX_SCF  1000
        EPS_SCF     1.0E-6
      &END OUTER_SCF
    &END SCF
    &QS
      EPS_DEFAULT     1.0000000000000000E-10
    &END QS
    &MGRID
      NGRIDS  4
      CUTOFF    600
      REL_CUTOFF   60
    &END MGRID
    &XC
      &XC_FUNCTIONAL  NO_SHORTCUT
        &PBE  T
        &END PBE
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        POTENTIAL_TYPE  PAIR_POTENTIAL
        &PAIR_POTENTIAL
          R_CUTOFF     1.6000000000000000E+01
          TYPE  DFTD3
          PARAMETER_FILE_NAME /home/johannes/sandbox/cp2k-5.1/data/dftd3.dat
          REFERENCE_FUNCTIONAL PBE
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
    @INCLUDE localize.inc
  &END DFT
  @INCLUDE subsys.inc
&END FORCE_EVAL
