PLEIADESROOT=/nobackup/eanders/convective_penetration_paper/dedalus_code/

# linear P runs
PLEIADESBASE=$PLEIADESROOT/paper_linear_P_cut/
LOCALBASE=linear_P_cut/
declare -a StringArray=(\
"linear_3D_Re8e2_P1e-2_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.023"
"linear_3D_Re8e2_P3e-2_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.04"
"linear_3D_Re8e2_P1e-1_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.073"
"linear_3D_Re8e2_P3e-1_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.126"
"linear_3D_Re8e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.23"
"linear_3D_Re8e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.325"
"linear_3D_Re8e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.46"
"linear_3D_Re8e2_P8e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.65"
"linear_3D_Re8e2_P1.6e1_zeta1e-3_S1e3_Lz2.25_Lcz1_Pr0.5_a2_Titer0_64x64x256_predictive0.92"
)
for d in ${StringArray[@]}
do
	echo $d
	LOCALDIR=$LOCALBASE/$d
	PLEIADESDIR=$PLEIADESBASE/$d
	mkdir $LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/cz_velocities.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/trace_top_cz.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/data_top_cz.h5 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/comparison_top_cz_trace.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/snapshots.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/profile_mov.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/avg_profs.tar ./$LOCALDIR/
done


# erf AE runs
PLEIADESBASE=$PLEIADESROOT/paper_erf_AE_cut/
LOCALBASE=erf_AE_cut/
declare -a StringArray=(\
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.4"
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.7"
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_schwarzschild_iters"
)
for d in ${StringArray[@]}
do
	echo $d
	LOCALDIR=$LOCALBASE/$d
	PLEIADESDIR=$PLEIADESBASE/$d
	mkdir $LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/cz_velocities.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/trace_top_cz.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/data_top_cz.h5 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/comparison_top_cz_trace.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/snapshots.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/profile_mov.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/avg_profs.tar ./$LOCALDIR/
done


# erf P runs
PLEIADESBASE=$PLEIADESROOT/paper_erf_P_cut/
LOCALBASE=erf_P_cut/
declare -a StringArray=(\
"erf_step_3D_Re4e2_P1e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.14"
"erf_step_3D_Re4e2_P2e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.275"
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.55"
"erf_step_3D_Re4e2_P6e0_zeta1e-3_S1e3_Lz3_Lcz1_Pr0.5_a2_Titer100_128x128x512_predictive0.83"
"erf_step_3D_Re4e2_P8e0_zeta1e-3_S1e3_Lz3_Lcz1_Pr0.5_a2_Titer100_128x128x512_predictive1.1"
"erf_step_3D_Re4e2_P1e1_zeta1e-3_S1e3_Lz3_Lcz1_Pr0.5_a2_Titer100_128x128x512_predictive1.38"
"erf_step_3D_Re4e2_P2e1_zeta1e-3_S1e3_Lz4.25_Lcz1_Pr0.5_a2_Titer100_128x128x512_predictive2.75"
)
for d in ${StringArray[@]}
do
	echo $d
	LOCALDIR=$LOCALBASE/$d
	PLEIADESDIR=$PLEIADESBASE/$d
	mkdir $LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/cz_velocities.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/trace_top_cz.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/data_top_cz.h5 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/comparison_top_cz_trace.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/snapshots.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/profile_mov.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/avg_profs.tar ./$LOCALDIR/
done

# erf Re runs
PLEIADESBASE=$PLEIADESROOT/paper_erf_Re_cut/
LOCALBASE=erf_Re_cut/
declare -a StringArray=(\
"erf_step_3D_Re1e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_32x32x256_predictive0.56"
"erf_step_3D_Re2.5e1_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_16x16x256_predictive0.44"
"erf_step_3D_Re2e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_64x64x256_predictive0.56"
"erf_step_3D_Re5e1_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_32x32x256_predictive0.49"
"erf_step_3D_Re8e2_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_128x128x256_predictive0.5"
"erf_step_3D_Re1.6e3_P4e0_zeta1e-3_S1e3_Lz2_Lcz1_Pr0.5_a2_Titer200_128x128x256_predictive0.485"
)
for d in ${StringArray[@]}
do
	echo $d
	LOCALDIR=$LOCALBASE/$d
	PLEIADESDIR=$PLEIADESBASE/$d
	mkdir $LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/cz_velocities.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/trace_top_cz.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/data_top_cz.h5 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/comparison_top_cz_trace.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/snapshots.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/profile_mov.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/avg_profs.tar ./$LOCALDIR/
done


# erf S runs
PLEIADESBASE=$PLEIADESROOT/paper_erf_S_cut/
LOCALBASE=erf_S_cut/
declare -a StringArray=(\
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S1e2_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.55"
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S3e2_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x256_predictive0.55"
"erf_step_3D_Re4e2_P4e0_zeta1e-3_S3e3_Lz2_Lcz1_Pr0.5_a2_Titer100_64x64x512_predictive0.55"
)
for d in ${StringArray[@]}
do
	echo $d
	LOCALDIR=$LOCALBASE/$d
	PLEIADESDIR=$PLEIADESBASE/$d
	mkdir $LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/cz_velocities.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/trace_top_cz.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/data_top_cz.h5 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/top_cz/comparison_top_cz_trace.png ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/snapshots.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/profile_mov.mp4 ./$LOCALDIR/
	scp pfe:$PLEIADESDIR/avg_profs.tar ./$LOCALDIR/
done


