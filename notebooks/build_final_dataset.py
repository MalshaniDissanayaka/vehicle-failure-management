import os, json, zipfile, textwrap, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

BASE='/mnt/data'
SEED=20261019
rng=np.random.default_rng(SEED)
N_SEQ=500
SEQ_LEN=100
N=N_SEQ*SEQ_LEN

# ---------- Load sources ----------
engine_path=os.path.join(BASE,'engine_data.csv')
obd_path=os.path.join(BASE,'carOBD_clean.csv')
engine=pd.read_csv(engine_path)
obd=pd.read_csv(obd_path)

# Engine Health source has generic engine readings; oil-pressure source units are documented as bar.
eng=engine.rename(columns={
    'Lub oil pressure':'oil_pressure_source_bar',
    'Coolant temp':'coolant_temp_c',
    'Engine rpm':'source_engine_rpm'
}).copy()
for c in ['oil_pressure_source_bar','coolant_temp_c']:
    eng[c]=pd.to_numeric(eng[c],errors='coerce')
eng_valid=eng.dropna(subset=['oil_pressure_source_bar','coolant_temp_c']).copy()
eng_valid=eng_valid[
    eng_valid['oil_pressure_source_bar'].between(0.05,8.0)
    & eng_valid['coolant_temp_c'].between(60,120)
].copy()
if len(eng_valid)<1000: raise RuntimeError('Not enough valid engine source rows')

# Filter OBD to usable live records and plausible numeric ranges. Preserve source order within file/runtime.
obd2=obd.copy()
for c in ['engine_run_time','engine_rpm','vehicle_speed','throttle','engine_load']:
    obd2[c]=pd.to_numeric(obd2[c],errors='coerce')
obd_live=obd2[obd2['operating_mode'].astype(str).str.lower().eq('live')].copy()
obd_live=obd_live.dropna(subset=['source_file','engine_run_time','engine_rpm','vehicle_speed','throttle','engine_load'])
obd_live=obd_live[
    obd_live['engine_rpm'].between(500,2500)
    & obd_live['vehicle_speed'].between(0,130)
    & obd_live['throttle'].between(0,100)
    & obd_live['engine_load'].between(0,100)
].copy()
if len(obd_live)<SEQ_LEN*10: raise RuntimeError('Not enough usable OBD source rows')

# Select 350 source-order blocks from OBD source; sequences are simulated blocks, not asserted continuous trips.
obd_blocks=[]
for sf,g in obd_live.groupby('source_file',sort=True):
    g=g.sort_values('engine_run_time',kind='stable').reset_index(drop=True)
    if len(g)>=SEQ_LEN:
        # candidate start points, spaced to avoid repeated overlapping windows within one source file
        starts=np.arange(0,len(g)-SEQ_LEN+1,SEQ_LEN)
        for st in starts:
            block=g.iloc[st:st+SEQ_LEN]
            if len(block)==SEQ_LEN:
                obd_blocks.append(block[['engine_rpm','vehicle_speed','throttle','engine_load']].to_numpy(float))
if len(obd_blocks)<350:
    # fallback random windows from filtered OBD rows; still disclosed as source-order blocks, not true journeys
    obd_blocks=[]
    for _ in range(500):
        sf=str(rng.choice(obd_live['source_file'].unique()))
        g=obd_live[obd_live['source_file']==sf].sort_values('engine_run_time',kind='stable').reset_index(drop=True)
        if len(g)>=SEQ_LEN:
            st=int(rng.integers(0,len(g)-SEQ_LEN+1))
            obd_blocks.append(g.iloc[st:st+SEQ_LEN][['engine_rpm','vehicle_speed','throttle','engine_load']].to_numpy(float))
if len(obd_blocks)<350: raise RuntimeError(f'Not enough OBD blocks: {len(obd_blocks)}')

# 200 normal + 150 overheating sequences use OBD blocks. Low-oil-pressure sequences use controlled stationary test points.
perm=rng.permutation(len(obd_blocks))
chosen=[obd_blocks[i] for i in perm[:350]]
scenario_names=np.array(['normal']*200+['overheating']*150+['low_oil_pressure']*150,dtype=object)
rng.shuffle(scenario_names)
# Build sequence IDs with scenario assignment and OBD blocks assigned to the 350 non-oil sequences.
seq_scenarios=scenario_names.tolist()
non_oil_ids=[i for i,s in enumerate(seq_scenarios) if s!='low_oil_pressure']
low_oil_ids=[i for i,s in enumerate(seq_scenarios) if s=='low_oil_pressure']
if len(non_oil_ids)!=350 or len(low_oil_ids)!=150: raise RuntimeError('Scenario allocation failure')

rows=[]
# Quantile mapping: preserve generic source pressure distribution shape, but map it to an RPM-dependent synthetic envelope.
# Toyota manual only provides specified test points; intermediate RPM interpolation is an explicit modelling assumption.
source_p=eng_valid['oil_pressure_source_bar'].to_numpy(float)
source_p_sorted=np.sort(source_p)
source_c=eng_valid['coolant_temp_c'].to_numpy(float)

def empirical_percentile(vals):
    # stable empirical ranks in [0,1]
    return np.searchsorted(source_p_sorted, vals, side='right') / len(source_p_sorted)

def pressure_envelope(rpm, q):
    # Reference anchor at idle: >=59 kPa; synthetic idle baseline upper cap 130 kPa (assumption).
    # At 2500 RPM: 140-280 kPa manual range. Linear interpolation between anchors is an assumption.
    rpm=np.asarray(rpm,float); q=np.asarray(q,float)
    low=np.interp(rpm,[750,2500],[65,140])
    high=np.interp(rpm,[750,2500],[130,270])
    return low + q*(high-low)

block_iter=iter(chosen)
for sid,scenario in enumerate(seq_scenarios, start=1):
    steps=np.arange(SEQ_LEN)
    if scenario in ('normal','overheating'):
        block=next(block_iter).copy()
        rpm=np.clip(block[:,0],500,2500)
        speed=np.clip(block[:,1],0,130)
        throttle=np.clip(block[:,2],0,100)
        load=np.clip(block[:,3],0,100)
        # Avoid an unsupported claim of exact temporal continuity; add small sensor-like correlated noise.
        for arr,sd,lo,hi in [(rpm,8,500,2500),(speed,0.35,0,130),(throttle,0.35,0,100),(load,0.8,0,100)]:
            eps=rng.normal(0,sd,SEQ_LEN)
            ar=np.zeros(SEQ_LEN)
            for j in range(1,SEQ_LEN): ar[j]=0.65*ar[j-1]+eps[j]
            arr += ar
            np.clip(arr,lo,hi,out=arr)
        # Sample one baseline coolant value per sequence from source distribution; add low autocorrelated sensor noise.
        base_pool=source_c[source_c<=95] if scenario=='overheating' else source_c
        base_temp=float(rng.choice(base_pool))
        temp=np.full(SEQ_LEN,base_temp,dtype=float)
        eps=rng.normal(0,0.25,SEQ_LEN); ar=np.zeros(SEQ_LEN)
        for j in range(1,SEQ_LEN): ar[j]=0.75*ar[j-1]+eps[j]
        temp += ar
        # Oil baseline pressure rank sampled from source data, mapped to an RPM-dependent envelope.
        sampled_bar=float(rng.choice(source_p))
        q=float(np.searchsorted(source_p_sorted,sampled_bar,side='right')/len(source_p_sorted))
        oil=pressure_envelope(rpm,np.full(SEQ_LEN,q))
        oil_noise=np.zeros(SEQ_LEN); eps=rng.normal(0,1.8,SEQ_LEN)
        for j in range(1,SEQ_LEN): oil_noise[j]=0.7*oil_noise[j-1]+eps[j]
        oil=np.clip(oil+oil_noise,pressure_envelope(rpm,np.zeros(SEQ_LEN)),pressure_envelope(rpm,np.ones(SEQ_LEN)))
        fault_type=np.array(['normal']*SEQ_LEN,dtype=object)
        if scenario=='overheating':
            onset=int(rng.integers(28,41))
            # Gradual thermal rise, with a smooth ramp and small correlated sensor noise.
            ramp=np.maximum(steps-onset,0)
            rate=float(rng.uniform(0.45,0.65))
            thermal_noise=np.zeros(SEQ_LEN); en=rng.normal(0,0.18,SEQ_LEN)
            for j in range(1,SEQ_LEN): thermal_noise[j]=0.8*thermal_noise[j-1]+en[j]
            temp += ramp*rate + thermal_noise
            fault_type[steps>=onset]='overheating'
    else:
        # Explicit Toyota manual test conditions: stationary idle (750 RPM) or 2500 RPM.
        test_rpm=int(rng.choice([750,2500]))
        rpm=np.full(SEQ_LEN,float(test_rpm))
        speed=np.zeros(SEQ_LEN)
        throttle=np.clip(rng.normal(2.0,0.5,SEQ_LEN),0,5)
        load=np.clip(rng.normal(18,2.0,SEQ_LEN),8,30)
        base_temp=float(rng.choice(source_c))
        temp=np.full(SEQ_LEN,base_temp,dtype=float)
        en=rng.normal(0,0.18,SEQ_LEN); tnoise=np.zeros(SEQ_LEN)
        for j in range(1,SEQ_LEN): tnoise[j]=0.75*tnoise[j-1]+en[j]
        temp+=tnoise
        # Start within a normal range for the relevant test point; decline gradually through warning band and below spec.
        onset=int(rng.integers(28,41))
        if test_rpm==750:
            threshold=59.0
            start=float(rng.uniform(82,105))
            end=float(rng.uniform(35,52))
        else:
            threshold=140.0
            start=float(rng.uniform(190,235))
            end=float(rng.uniform(85,125))
        oil=np.full(SEQ_LEN,start,dtype=float)
        ramp=np.clip((steps-onset)/float(rng.integers(24,34)),0,1)
        oil=start+(end-start)*ramp
        onoise=np.zeros(SEQ_LEN); en=rng.normal(0,0.8,SEQ_LEN)
        for j in range(1,SEQ_LEN): onoise[j]=0.65*onoise[j-1]+en[j]
        oil+=onoise
        fault_type=np.array(['normal']*SEQ_LEN,dtype=object)
        fault_type[steps>=onset]='low_oil_pressure'
    # Store unrounded raw values for all sequences; labels assigned only after rounding.
    rows.append(pd.DataFrame({
        'sequence_id':sid,
        'timestamp_step':steps,
        'engine_coolant_temperature':temp,
        'oil_pressure_kpa':oil,
        'engine_rpm':rpm,
        'vehicle_speed':speed,
        'throttle_position':throttle,
        'engine_load':load,
        'scenario_type':scenario,
        'fault_type':fault_type,
    }))

df=pd.concat(rows,ignore_index=True)
# Round values before assigning labels, so saved values and labels agree.
round_cols=['engine_coolant_temperature','oil_pressure_kpa','engine_rpm','vehicle_speed','throttle_position','engine_load']
df[round_cols]=df[round_cols].round(2)

# Severity labels. Coolant rules are declared research modelling assumptions, not Toyota official thresholds.
df['condition']='normal'
overheat_active=df['fault_type'].eq('overheating')
df.loc[overheat_active & (df.engine_coolant_temperature>100),'condition']='warning' # assumed warning threshold
df.loc[overheat_active & (df.engine_coolant_temperature>135),'condition']='critical' # diagnostic-reference-based model rule, not standalone proof

# Oil pressure labels apply only to the controlled manual test-point scenarios and only after onset.
oil_active=df['fault_type'].eq('low_oil_pressure')
idle_test=oil_active & (df.engine_rpm==750) & (df.vehicle_speed==0)
rpm2500_test=oil_active & (df.engine_rpm==2500) & (df.vehicle_speed==0)
# Warning band is a declared 15% margin assumption; critical threshold follows the manual test point.
for mask,threshold in [(idle_test,59.0),(rpm2500_test,140.0)]:
    df.loc[mask & (df.oil_pressure_kpa < threshold*1.15) & (df.oil_pressure_kpa >= threshold),'condition']='warning'
    df.loc[mask & (df.oil_pressure_kpa < threshold),'condition']='critical'
# fault_active is onset flag, not severity label
df['fault_active']=(df['fault_type']!='normal').astype('int8')

# Stable column order
cols=['sequence_id','timestamp_step','engine_coolant_temperature','oil_pressure_kpa','engine_rpm','vehicle_speed','throttle_position','engine_load','scenario_type','fault_type','condition','fault_active']
df=df[cols]

# ---------- Validation ----------
checks={}
def ck(name, val): checks[name]=bool(val)
ck('exactly_50000_rows',len(df)==50000)
ck('exactly_12_columns',df.shape[1]==12)
ck('no_missing_values',int(df.isna().sum().sum())==0)
ck('no_exact_duplicate_rows',not df.duplicated().any())
ck('exactly_500_sequences',df.sequence_id.nunique()==500)
ck('100_rows_per_sequence',(df.groupby('sequence_id').size()==100).all())
ck('steps_0_to_99_each_sequence',(df.groupby('sequence_id').timestamp_step.apply(lambda x: list(x)==list(range(100))).all()))
ck('valid_scenario_labels',set(df.scenario_type.unique())=={'normal','overheating','low_oil_pressure'})
ck('valid_fault_labels',set(df.fault_type.unique())=={'normal','overheating','low_oil_pressure'})
ck('valid_condition_labels',set(df.condition.unique()).issubset({'normal','warning','critical'}))
ck('fault_active_matches_fault_type',((df.fault_type!='normal').astype(int)==df.fault_active).all())
ck('scenario_constant_within_sequence',(df.groupby('sequence_id').scenario_type.nunique()==1).all())
ck('normal_scenario_has_no_fault',(df.loc[df.scenario_type=='normal','fault_type']=='normal').all())
ck('fault_type_matches_scenario_after_onset',all(((g.loc[g.fault_type!='normal','fault_type']==g.scenario_type.loc[g.fault_type!='normal']).all()) for _,g in df.groupby('sequence_id')))
ck('finite_numeric_features',np.isfinite(df[round_cols].to_numpy()).all())
ck('coolant_range_60_150',df.engine_coolant_temperature.between(60,150).all())
ck('oil_pressure_positive',df.oil_pressure_kpa.gt(0).all())
# Normal and overheating sequences use the declared RPM-conditioned synthetic envelope.
_non_oil=df[df.scenario_type.isin(['normal','overheating'])]
_env_lo=np.interp(_non_oil.engine_rpm.to_numpy(float),[750,2500],[65,140])
_env_hi=np.interp(_non_oil.engine_rpm.to_numpy(float),[750,2500],[130,270])
ck('baseline_oil_pressure_within_declared_rpm_envelope',((_non_oil.oil_pressure_kpa.to_numpy(float)>=_env_lo-0.02)&(_non_oil.oil_pressure_kpa.to_numpy(float)<=_env_hi+0.02)).all())
ck('rpm_range_500_2500',df.engine_rpm.between(500,2500).all())
ck('speed_range_0_130',df.vehicle_speed.between(0,130).all())
ck('throttle_range_0_100',df.throttle_position.between(0,100).all())
ck('load_range_0_100',df.engine_load.between(0,100).all())
ck('overheating_warning_consistency',not ((df.fault_type=='overheating')&(df.condition=='warning')&(df.engine_coolant_temperature<=100)).any())
ck('overheating_critical_consistency',not ((df.fault_type=='overheating')&(df.condition=='critical')&(df.engine_coolant_temperature<=135)).any())
ck('oil_critical_idle_below_59',not (idle_test & (df.condition=='critical') & (df.oil_pressure_kpa>=59)).any())
ck('oil_critical_2500_below_140',not (rpm2500_test & (df.condition=='critical') & (df.oil_pressure_kpa>=140)).any())
ck('oil_warning_idle_band',not (idle_test & (df.condition=='warning') & ~df.oil_pressure_kpa.between(59,59*1.15,inclusive='left')).any())
ck('oil_warning_2500_band',not (rpm2500_test & (df.condition=='warning') & ~df.oil_pressure_kpa.between(140,140*1.15,inclusive='left')).any())
ck('oil_fault_sequences_are_stationary_test_conditions',df.loc[df.scenario_type=='low_oil_pressure','vehicle_speed'].eq(0).all() and df.loc[df.scenario_type=='low_oil_pressure','engine_rpm'].isin([750,2500]).all())
ck('faults_have_pre_onset_normal_rows',all((g.iloc[:np.where(g.fault_type.to_numpy()!='normal')[0][0]].condition=='normal').all() for _,g in df[df.scenario_type!='normal'].groupby('sequence_id') if (g.fault_type!='normal').any()))
# For gradual direction, compare pre-onset and late post-onset median by sequence.
gradual_overheat=[]; gradual_oil=[]
for sid,g in df.groupby('sequence_id'):
    s=g.scenario_type.iloc[0]
    act=np.where(g.fault_type.to_numpy()!='normal')[0]
    if not len(act): continue
    onset=act[0]
    if s=='overheating': gradual_overheat.append(g.engine_coolant_temperature.iloc[-1] > g.engine_coolant_temperature.iloc[onset]+15)
    elif s=='low_oil_pressure': gradual_oil.append(g.oil_pressure_kpa.iloc[-1] < g.oil_pressure_kpa.iloc[onset]-15)
ck('overheating_sequences_show_gradual_temperature_rise',all(gradual_overheat) and len(gradual_overheat)==150)
ck('oil_fault_sequences_show_gradual_pressure_drop',all(gradual_oil) and len(gradual_oil)==150)
# No fault-active or scenario fields accidentally in sensor features
ck('six_sensor_features_present',set(round_cols)=={'engine_coolant_temperature','oil_pressure_kpa','engine_rpm','vehicle_speed','throttle_position','engine_load'})

# Compare generated baseline scenario (normal) to filtered source distributions.
normal_df=df[df.scenario_type=='normal']
source_obd=obd_live
# Baseline features: normal scenario rows only; RPM/speed/throttle/load directly sampled from OBD source blocks.
source_engine=eng_valid

def stats(series):
    s=pd.Series(series).dropna()
    return {'n':int(s.size),'mean':round(float(s.mean()),3),'std':round(float(s.std()),3),'q05':round(float(s.quantile(.05)),3),'median':round(float(s.median()),3),'q95':round(float(s.quantile(.95)),3),'min':round(float(s.min()),3),'max':round(float(s.max()),3)}
comparisons={}
# Coolant compares direct to source coolant distribution, as baseline source values are sampled directly.
comparisons['engine_coolant_temperature']={'generated_normal_scenario':stats(normal_df.engine_coolant_temperature),'filtered_source_reference':stats(source_engine.coolant_temp_c)}
for gen_col,src_col in [('engine_rpm','engine_rpm'),('vehicle_speed','vehicle_speed'),('throttle_position','throttle'),('engine_load','engine_load')]:
    comparisons[gen_col]={'generated_normal_scenario':stats(normal_df[gen_col]),'filtered_source_reference':stats(source_obd[src_col])}
# Oil source is generic and its kPa-converted raw scale does not directly match Toyota reference points; compare rank shape only.
source_p_kpa=source_engine.oil_pressure_source_bar*100
# percentile rank distribution of generated baseline pressure within its RPM-conditioned synthetic envelope vs source pressure percentile distribution.
source_q=np.searchsorted(np.sort(source_engine.oil_pressure_source_bar.to_numpy()),source_engine.oil_pressure_source_bar.to_numpy(),side='right')/len(source_engine)
gen_oil=normal_df.oil_pressure_kpa.to_numpy()
# The direct pressure values are intentionally adapted to manual reference anchors; don't claim direct raw-unit distribution equality.
comparisons['oil_pressure_kpa']={
 'generated_normal_scenario':stats(gen_oil),
 'source_raw_bar_converted_to_kpa_reference':stats(source_p_kpa),
 'interpretation':'Not a direct distribution match: generic engine source values are converted to kPa but mapped to an RPM-conditioned synthetic envelope because raw generic-engine pressures do not match Toyota Prius c manual test-point specifications. Percentile shape is used only as a prior.'
}
# Compare percentile shape after mapping to the declared RPM-dependent envelope (not raw kPa).
normal_rpm=normal_df.engine_rpm.to_numpy(float)
normal_pressure=normal_df.oil_pressure_kpa.to_numpy(float)
lo_env=np.interp(normal_rpm,[750,2500],[65,140])
hi_env=np.interp(normal_rpm,[750,2500],[130,270])
normalized_oil_rank=np.clip((normal_pressure-lo_env)/(hi_env-lo_env),0,1)
source_rank=np.searchsorted(np.sort(source_engine.oil_pressure_source_bar.to_numpy()),source_engine.oil_pressure_source_bar.to_numpy(),side='right')/len(source_engine)
comparisons['oil_pressure_percentile_shape']={
 'generated_normal_normalized_position_in_synthetic_envelope':stats(normalized_oil_rank),
 'source_engine_oil_pressure_empirical_percentile':stats(source_rank),
 'interpretation':'Percentile-shape comparison only. The RPM envelope is an explicit modelling transformation, not a Toyota specification at intermediate RPM.'
}
# Calculate mean absolute standardized differences for direct-comparable baseline features.
direct_features=['engine_coolant_temperature','engine_rpm','vehicle_speed','throttle_position','engine_load']
std_diff={}
for c in direct_features:
    gen=normal_df[c]
    src={'engine_coolant_temperature':source_engine.coolant_temp_c,'engine_rpm':source_obd.engine_rpm,'vehicle_speed':source_obd.vehicle_speed,'throttle_position':source_obd.throttle,'engine_load':source_obd.engine_load}[c]
    std=max(float(pd.Series(src).std()),1e-9)
    std_diff[c]=round(abs(float(gen.mean())-float(pd.Series(src).mean()))/std,3)

# Distribution similarity check is informative, not an engineering proof. Threshold chosen as <0.25 source SD for means; quantiles are also reported.
checks['baseline_means_within_0_25_source_sd']=all(v<0.25 for v in std_diff.values())
# Ensure oil test point baseline values respect manual lower reference before fault onset; values may have sensor noise.
pre_oil=df[(df.scenario_type=='low_oil_pressure')&(df.fault_type=='normal')]
checks['oil_test_baseline_above_manual_lower_limit']=bool(
    ((pre_oil.loc[pre_oil.engine_rpm==750,'oil_pressure_kpa']>=59).all()) and
    ((pre_oil.loc[pre_oil.engine_rpm==2500,'oil_pressure_kpa']>=140).all())
)
# For 2500 RPM test point normal baseline should not exceed 280 kPa manual range.
checks['oil_2500_test_baseline_within_manual_range']=bool(pre_oil.loc[pre_oil.engine_rpm==2500,'oil_pressure_kpa'].between(140,280).all())

# Update check summary after all checks
passed=sum(checks.values()); total=len(checks)

# Save dataset
csv_name='FYP_synthetic_vehicle_fault_dataset_50000_checked.csv'
csv_path=os.path.join(BASE,csv_name)
df.to_csv(csv_path,index=False,float_format='%.2f')
# Re-read exact saved artifact and re-run key persisted checks
saved=pd.read_csv(csv_path)
persisted_checks={
    'saved_csv_50000_rows':len(saved)==50000,
    'saved_csv_no_missing':int(saved.isna().sum().sum())==0,
    'saved_csv_no_duplicates':not saved.duplicated().any(),
    'saved_csv_sequence_integrity':saved.groupby('sequence_id').size().eq(100).all(),
    'saved_csv_fault_active_consistency':((saved.fault_type!='normal').astype(int)==saved.fault_active).all(),
}
checks.update(persisted_checks)
passed=sum(checks.values()); total=len(checks)

# Build report with clear separation of structural pass vs scientific limitations.
report={
 'title':'FYP Synthetic Vehicle Fault Dataset — Audit and Validation Report',
 'generated_utc':datetime.now(timezone.utc).isoformat(),
 'seed':SEED,
 'source_files':{'engine':'engine_data.csv','obd':'carOBD_clean.csv','proposal':'Group 18_FYP Project Proposal .pdf'},
 'dataset_file':csv_name,
 'rows':len(saved),'columns':list(saved.columns),'feature_columns':round_cols,
 'sequence_count':int(saved.sequence_id.nunique()),
 'rows_per_sequence':{'min':int(saved.groupby('sequence_id').size().min()),'max':int(saved.groupby('sequence_id').size().max())},
 'scenario_sequence_counts':{str(k):int(v) for k,v in saved.groupby('sequence_id').scenario_type.first().value_counts().items()},
 'scenario_row_counts':{str(k):int(v) for k,v in saved.scenario_type.value_counts().items()},
 'fault_type_counts':{str(k):int(v) for k,v in saved.fault_type.value_counts().items()},
 'condition_counts':{str(k):int(v) for k,v in saved.condition.value_counts().items()},
 'condition_percentages':{str(k):round(float(v/len(saved)*100),2) for k,v in saved.condition.value_counts().items()},
 'missing_values_total':int(saved.isna().sum().sum()),
 'exact_duplicate_rows':int(saved.duplicated().sum()),
 'checks':checks,'checks_passed':passed,'checks_total':total,'all_checks_pass':all(checks.values()),
 'source_filtering':{
  'engine_source_rows_original':int(len(engine)), 'engine_source_rows_used_for_baseline':int(len(eng_valid)),
  'engine_filter':'nonmissing source oil pressure and coolant; source pressure 0.05–8.0 bar; coolant 60–120 °C. Extreme coolant outliers excluded from baseline sampling.',
  'obd_source_rows_original':int(len(obd)), 'obd_live_rows_used':int(len(obd_live)),
  'obd_filter':'operating_mode=live; RPM 500–2500; speed 0–130; throttle/load 0–100; nonmissing engine_run_time.',
  'sequence_caveat':'OBD blocks are source-order blocks sorted by source_file and engine_run_time after filtering. Filtering/repeated runtime means they are simulated sequences, not guaranteed continuous real trips.'},
 'distribution_comparison':comparisons,
 'baseline_mean_difference_in_source_sd':std_diff,
 'fault_logic':{
  'overheating':'Fault onset is gradual after a sequence-specific onset. condition=warning above 100°C and critical above 135°C only within overheating scenario. These are documented modelling assumptions; >135°C in the manual is diagnostic context, not standalone proof of overheating.',
  'low_oil_pressure':'Generated only under stationary controlled test points: 750 RPM idle and 2500 RPM. Manual references: >=59 kPa at idle; 140–280 kPa at 2500 RPM. Warning band = up to 15% above the lower spec (research assumption); critical below lower spec. Pressure declines gradually.',
  'normal_oil_pressure':'Generic engine source pressure percentile ranks (source values documented in bar) are mapped to an RPM-conditioned synthetic kPa envelope; raw values are not treated as Toyota kPa readings. Interpolation/upper bounds at non-manual RPM values are assumptions, not Toyota specifications.'},
 'manual_references':[
  {'topic':'Toyota Prius c oil pressure inspection','url':'https://lemon.dogeware.me/Toyota/2016/Prius%20C%20One/Repair%20and%20Diagnosis%20%28Single%20Page%29/Engine%20Mechanical/Lubrication%20System/Engine%20Lubrication%20System%20%28Inspection%29/Lubrication%20System/On-Vehicle%20Inspection%20%5B11%2F2015%20-%20%5D/Procedure/','reference':'At idle minimum 59 kPa; at 2500 RPM 140–280 kPa, under stated test conditions.'},
  {'topic':'Toyota Prius c coolant temperature diagnostics','url':'https://lemon.dogeware.me/Toyota/2016/Prius%20C%20One/Repair%20and%20Diagnosis%20%28Single%20Page%29/Engine%20Performance/Testing%20%26%20Diagnosis/SFI%20System%20-%20Diagnostics%20-%20Introduction%20%282%20Of%202%29/SFI%20System/Data%20List%20%2F%20Active%20Test%20%5B06%2F2017%20-%20%5D/','reference':'Warm engine 75–100°C; >135°C can indicate circuit/sensor issue and requires overheating check; not a standalone fault threshold.'},
  {'topic':'Engine source dataset card','url':'https://huggingface.co/datasets/dhani10/engine-condition-dataset','reference':'Used to document source unit and feature provenance; source dataset is not Toyota Aqua-specific.'}
 ],
 'limitations':[
  'Synthetic dataset; no actual Toyota Aqua fault records used for ground-truth validation.',
  'Coolant warning (>100°C), critical (>135°C), 15% oil warning margin, and RPM interpolation are modelling assumptions; they are not all manufacturer-declared thresholds.',
  'Toyota Prius c manual is used as a proxy reference for Toyota Aqua; exact model year/engine applicability should be checked.',
  'Source distributions are statistical priors, not proof of diagnostic validity. Oil pressure generic-source values are not directly matched to Toyota manual kPa range; their rank shape is mapped to a synthetic RPM envelope.',
  'This validates internal consistency and source similarity, not real-world diagnostic accuracy.'
 ],
 'recommended_ml_use':{
  'predictor_features':round_cols,
  'targets':['fault_type','condition'],
  'do_not_use_as_predictors':['sequence_id','timestamp_step','scenario_type','fault_active'],
  'split':'Train/test split by sequence_id to prevent records from the same simulated sequence leaking across splits.'
 }
}
report_path=os.path.join(BASE,'FYP_synthetic_dataset_checked_validation_report.json')
with open(report_path,'w') as f: json.dump(report,f,indent=2,default=lambda x: x.item() if hasattr(x, 'item') else str(x))

# Methodology CSV
method_rows=[
 ('Purpose','50,000-row synthetic research dataset for the FYP; target fault categories are engine overheating and low engine oil pressure.'),
 ('Features','Six features: coolant temperature (°C), oil pressure (kPa), engine RPM, vehicle speed, throttle position (%), engine load (%).'),
 ('Engine source','engine_data.csv; generic Engine Health data. Source oil pressure documented in bar; conversion to kPa is ×100, but converted generic readings are not assumed to be Toyota pressure measurements.'),
 ('OBD source','carOBD_clean.csv; live-mode records filtered to plausible RPM/speed/throttle/load. Source-order blocks provide driving-feature patterns; not asserted as verified continuous trips.'),
 ('Dataset structure','50,000 rows in 500 sequences × 100 steps; 200 normal, 150 overheating, 150 low-oil-pressure sequences.'),
 ('Coolant baseline','Sampled from source coolant distribution (60–120°C after filtering), with small autocorrelated sensor noise.'),
 ('Oil pressure baseline','Generic source oil-pressure percentile ranks (source values documented in bar) are mapped to an RPM-conditioned synthetic kPa envelope. Raw source values are not treated as Toyota kPa readings; intermediate-RPM interpolation and bounds are modelling assumptions.'),
 ('Toyota oil pressure reference','Prius c manual: at idle at least 59 kPa; at 2500 RPM 140–280 kPa. These are applied only in stationary controlled test-point sequences.'),
 ('Low oil fault injection','Stationary test at 750 RPM or 2500 RPM; pressure declines gradually after onset. Critical below the corresponding lower manual limit. Warning band is 15% above the lower limit; 15% is a research assumption.'),
 ('Coolant references','Warm-engine reference 75–100°C. >135°C is a diagnostic context that may indicate sensor/circuit issue and requires overheating check; it is not standalone proof of overheating.'),
 ('Coolant labels','Warning >100°C and critical >135°C within injected overheating sequences are modelling rules, not Toyota official warning/critical thresholds.'),
 ('Rounding and labels','Features are rounded before severity labels are assigned so saved values agree with labels.'),
 ('Leakage prevention','Do not use sequence_id, timestamp_step, scenario_type or fault_active as predictor features. Split train/test by sequence_id.'),
 ('Validation','Structural checks, source-distribution comparison for directly comparable features, RPM/test-condition checks, gradual-onset checks, and saved-CSV re-read checks are recorded in JSON report.'),
 ('Limitations','Synthetic only; not validated on real Toyota Aqua fault logs. Source similarity and internal consistency do not prove real-world diagnostic accuracy.'),
]
method_path=os.path.join(BASE,'FYP_synthetic_dataset_checked_methodology.csv')
pd.DataFrame(method_rows,columns=['item','description']).to_csv(method_path,index=False)

readme=f'''FYP SYNTHETIC VEHICLE FAULT DATASET\n\nFiles\n- FYP_synthetic_vehicle_fault_dataset_50000_checked.csv: 50,000 rows.\n- FYP_synthetic_dataset_checked_methodology.csv: source, assumptions, and rules.\n- FYP_synthetic_dataset_checked_validation_report.json: validation checks and comparisons.\n\nScope\nTwo fault categories: engine overheating and low engine oil pressure. Labels: normal, warning, critical.\n\nUse\nUse the six sensor/vehicle features as predictors. Do not use sequence_id, timestamp_step, scenario_type, or fault_active as predictors. Split train/test by sequence_id. Choose either fault_type or condition as target.\n\nImportant scientific note\nThis is synthetic research data. It is checked for internal consistency and baseline similarity, but has not been validated against real Toyota Aqua fault logs. Some thresholds are explicit modelling assumptions rather than official Toyota warning/critical limits. Read the methodology CSV and JSON validation report before reporting results.\n\nValidation checks passed: {passed}/{total}.\n'''
readme_path=os.path.join(BASE,'README_FYP_synthetic_dataset_checked.txt')
with open(readme_path,'w') as f:f.write(readme)

# Package
zip_path=os.path.join(BASE,'FYP_synthetic_vehicle_fault_dataset_checked_package.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in [csv_path,method_path,report_path,readme_path]: z.write(p,arcname=os.path.basename(p))

# Print audit summary
print(json.dumps({
 'dataset_path':csv_path,'methodology_path':method_path,'report_path':report_path,'zip_path':zip_path,
 'rows':len(saved),'columns':len(saved.columns),'checks_passed':passed,'checks_total':total,
 'all_checks_pass':all(checks.values()),'failed_checks':[k for k,v in checks.items() if not v],
 'condition_counts':saved.condition.value_counts().to_dict(),
 'fault_counts':saved.fault_type.value_counts().to_dict(),
 'scenario_counts':saved.scenario_type.value_counts().to_dict(),
 'mean_diff_sd':std_diff,
 'comparison':comparisons,
 'pressure_test_pre_onset':pre_oil.groupby('engine_rpm').oil_pressure_kpa.agg(['min','median','max']).round(2).to_dict('index'),
 'fault_post_changes':{'overheat_all_sequences_rise_gt15':all(gradual_overheat),'oil_all_sequences_drop_gt15':all(gradual_oil)}
},indent=2,default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
