# Current goal handoff

## NEWEST: smallTP PD227 andDCP2994 installed; final1800 search running

HEAD82158e09. New stage data/model guard removal and evidence ready to commit;checkpoint61 pending.
PD50curves/227points all span8192, canonical native validation passed. DCP304newcases allnumericpass,
303stable,1CVexcluded;total2994nativequeries exact0. Source newDCP SHA0807f07623c0907c7c264de217607318e281f9a7efbda7f61b71063af453625c.
Frozen small-tp-pd-{primary,supplement,selected,merged}-20261025 andsmall-tp-dcp-{published,merged}-20261025.
41PD/DCPcampaigns archived; exact486filelist /tmp/aisim-small-pd-dcp-stage-files.json. Never stage raw originals wholesale.
TP2/DCP2 global32DeepEP/tensor each8x8k1kpasses;global16bothreject atsample.86memory;frozen small-tp-dcp-validation-20261025.
Previous36DCPpasses ->38now;model15tests pass,fullRust2051pass1ignored. No newRustchanges since a1859b5b native.
132aggregated search passed, frozenautomatic-aggregate-132-20261025.
ACTIVEroot24914 final1800 jointsearch,CPU/tmp/aisim-final-search-1800-20261025,hostlog/tmp/aisim-final-search-1800.log.
No NPU collection active;32and19freeofourwork,31untouched. Root93967/72844localavailable,89706remote19idle.
NEXTcommitthisdatastage;buildwheel fromcommittedsource+data,replaceoldimage190684,validateinstalledcommands,
freeze1800completeandbest-replay,updateusage/progress+checkpoint+finalcommit. Goalnotcompleteuntilruntimeupdated.

## NEWEST: DCP published36layouts pass; smallTPcompute complete, PD160 running

HEADdb54712ed888dc2b05e89e717391a695e729c438 commitsDCPsource+ADP12/evidence. Checkpoint59;nextpatch60prepared.
DCP2691table INSTALLED;exactnative2691querieszero;SHAa67e660e73368d3c241a06098c8bea2023800b0e41c0f0370fb1c110c681c3e4.
Frozenproduction/dcp-storage-release-20261025 anddcp-prediction-validation-20261025 (36strict8k1klayouts8requestseach).
390GPUgoldens pass486s;46plumbing/14model/9producer/3DCPnativepassed. Olduserimage190684...NOTupdated.
SmallTP1/2 compute DONE:linear468->462stable;vector160->160;BMM148->146;attention886->885numericpasses->870stable.
Rawdirs small-tp-{linear,vector}-node32-20261025,small-tp-{attention,bmm}-node19-20261025;controls small-tp-linear-controls-node32-20261025.
Newlinear16copies/16calls boundedmemory(TP1lmhead~1.9GB/copy);8vs16M8max3.028%median0.823%,16repeatone13.05%drift retained.
Oldmatrixcurvesretain64-copyprotocol;newcurvesdisjointonly. Attentionfailureindex99,h64prefill512prefix8192,onezeroelement out32768,
notpatched;rawfailurepreserved,excluded. /tmp/aisim-small-tp-compute-stage.json selects108archivedfiles.
Newplans production/small-tp-{linear,vector,bmm,attention,pd}-plan-20261025.json;fullsourceplans have.full.json.
Runtimeidentities small-tp-{linear,vector,bmm,attention}-runtime-identity-20261025.json generated.
UNCOMMITTED:profileTP1/2;nativeBMM/attention/vectorheads32/64;BMM/attentionidentitypergrid;curve_append supportsattention/BMM/PD
andpreservesexistingv2history;plannernewcurves supportsallfamilies andexplicitlarge-linearprotocoloverride.
ModelnowacceptslocalTP1/2,butdefaultcatalogSTILL12TP4/8/16shapes untilsmalldata/PDqualified.
collective(ranks1) returnsNone,filteredbeforecompile. TP1DeepEPfinalnormfused:Layoutomitsunitaxes;bind_exitkeep_output;
noextra finalresidualadd/noone-rankcomm. Sourceproofboundary.py688 andlayout.pysharded_over;logitshelperiscalledalsoTP1.
39Ascendnative,15producer,19model/appendtests passed. Root93967 currentlyreleasebuild /tmp/aisim-small-tp-release-build.log;
whencompleteatomiccopytoCPU/tmp/aisim-native. Currentloadednative18c93522...(DCP)stillrejectsnewheads atquery.
NEXTcommitvalidatedsmallproducer/consumer source +DCPdata/validation/docs;export4compute datasets withthatref;
appendtoexistingcurveswhilepreservingDCPattentiontableandhistory;queryallnative;memoryfilter/8k1kTP1/2probes.
ACTIVEroot72844node32 and89706node19:run_pd_grid suffixsmalltp-20261025,160cases/16newmappingsTP1/2 cross1/2/4/8/16.
Logs/tmp/aisim-small-tp-pd-node32.log andnode19.log;progresspd-grid-npu1-*-smalltp-20261025.json;only3mappingsdone atlastcheck.
DoNOTlaunchNPUworkwhilePDactive.31untouched. PDsourcea118...unchanged,mean8callprotocol.
PDnewcurvesdisjointold81;curve_append--operationpd supportsrecursiveoldcombinedprovenance;check8kcoverageforallnewcurves.
Remaining:finishsmallTP+PD,defaultcatalogmemoryfeasible expansion/search,finalwheel/image/installchecks/docscommits.
GPUrecommend explicitlydoesNOTsearchCP/DCP>1(config/engine.py);DCPpredict-only matchesbaseline. GPUReplayPP1only.
AscendFuseEPalsoforcesepsize=globalTP(_A2A_EP_SPANNING_BACKENDS),soitcannotrestorehybridMoETP/EPnonebackendfailure.
DoNOTclosegoalbeforefinalusableimage. Userrequestsfunctionalityfirst;servingaccuracyfuturebutallqualitygatesretained.

## NEWEST: DCP2736complete2691qualified; source integration tested; publish next

HEAD23412383 still; largeUNCOMMITTED source/evidence stage needscommit. Checkpoint58;patchcheckpoint-dcp-collection-stage.json prepared/applying59.
Frozen12-layout1296/1296search andADPvalidation; /tmp/aisim-adp-stage-files.json lists1614selectedfiles.
DCPcorrectFINALcontract=sglang_npu_dcp_storage_attention_chain/v1;plan dcp-storage-attention-plan-20261025.json.
CollectorSHA b257fed83ab1dffed40de7e8502d61224221a100bd1c6e509eb59e6eff7821f1 MUSTNOTchange/formatwhilepublishing.
Itusespage128ownership andincludesindexerKVwrite(onlyremaptrue),latent/ropeKVwrites,QAG,partialSA,FP32alltoall/update.
ALL2736attempted;2691qualified,37CVexcluded,8numericexcluded. Fullvalidatorpassed CPU/tmp/aisim-dcp-storage-validation.json,
frozenhostproduction/dcp-full-plan-validation-20261025.json. NoDCPparquetpublishedyet.
Campaigninputs: dcp-storage-world2-node19-20261025-v2 (912/1numericfail),world4-node19-v2(912/0),
dcp-storage-world16-node32-v2(304/2),world8-node32-v2(FAILEDthresholdprefix29/3),
world8-node32-v2-continuation(COMPLETED579/2). Fullnamesinclude20261025.
World8originalinvestigation.json authorizesreadingitsFAILEDprefixonlywithcompleteuntouchedcontinuationcoverage.
Numericfailures retained/excluded, NOTretriedorpassedbylooseningtolerance. ReferencechecksBF16partials+FP32LSE andmergedoutput;
controlledprobabilityprobesrejectsimpleBF16probability/exponentialhypotheses;noexactkernelprecisionclaim.
Runnernowrecordsper-attemptobs andadvancesPASTfailedcases,stopsaround10%/3consecutive;--start-indexexplicitcontinuation.
Exporterhandlesgziprawobservations,requiresallplanindicesonceandrecordedinvestigationforstoppedprefix.
/tmp/aisim-dcp-stage-files.json lists354archivedevidencefiles;oldwronginterleave1dataDIAGNOSTICONLY.
Newsource asc_dcp_attention.rs table/op,Opwiring,sharedPlane helper,canonicalPD/DCPguards,logicalblockguard.
Native18c93522e474309a1c08717e893c93b42cac9d9b3b5badd45b6afdd9c6a56337 loadedCPU/tmp/aisim-native (schema29appendonly).
3nativeDCPtests,9producer,14model,46GPU/NPU_DCPplumbingtests pass. GPU390parityRUNNINGroot24914log/tmp/aisim-dcp-gpu-parity.log.
ModelDCPomits3oldcachewriteops,uses21full+57reusechain;Lightningindexer/GEMM/BMMstayseparate.
KVindexerfullperrank;rankrate64896atDCP2,sentinel16613376,observed3,239,608,320bytes->49664logicaltokens.
Replayblock_size MUST128*dcp (e.g256) forphysical128pages;newexampleascend910c-glm52-dcp2-8k1k.yaml.
NPU DCP canonicalworker_typeaggregatedonly. Eagerdecodeinmixedforwardexplicitmissingdata (notfakegraphpricing).
supports_dcp_on_system preservesGPU.SGLangDSAstillrejectsbeforeconstruction;46old/newDCPtests pass.
IMPORTANTGPUrecommend rejectsCP/DCP>1 too (config/engine.pycustompresetscomment);thisisnotNPU-specificlack.
NextfinishRuff/sourcecommit inclfrozenplans/proofs,exportwithcommitref,newDCPtable/nativequery,then8k1kDCPconfigs.
RemainingafterDCP:smallerTP1/2primitivegridsandPDmappingsifmemory-feasible,finalwheel/image/docscommits. GoalACTIVE.
SmallTPlinearweights needboundedrotation:TP1lmhead~1.9GB*64copiesOOM;newprotocol16copies16iterationspreserveslargeworking-set,
mustrunmeaningfulcachecontrols. BMM/attentionnewheads32/64needactualcollection;donotjustrelaxguards.
AllownedNPUservers/jobs STOPPEDnow;32+19available,31untouched. Root89706remote19free,72844localfree,93967localfree,24914GPUparity.

## CURRENT ACTIVE: twelve layouts1296pass; DCP page128 audit corrected, no DCP data published

HEAD2341238367a5d02182c3987a254dd3b9ee7e9d77 publishes seven-layoutstage. Checkpoint57 integrity[].
UNCOMMITTED model now supports tensorMoEglobal16/32 EP1 with localTP4/8/16 and ADP=global/local;
defaultcatalog12 shapes andexamplemax_trials1296. CPU1296sweep COMPLETED, checkcounts/freeze next.
12modeltests pass;DP2real8k/16out andDP4real8k/16out passed, bothownedservers STOPPED.
ADP2resident50,480,189,948;ADP4resident52,840,002,044. Context:embeddingARlocalTP1 +156ARglobal16 +logitsAGlocalTP1.
Decode:embeddingARlocalTP1 +78RSlocalTP +78AGglobal16 +78RSglobal16 +79AGlocalTP(logitsincluded).
FFNpostnormuses_dp_gather_sum_read:localresidualaddbeforegather,plainRMSonGLOBALtokens;
shared/router/tensorMoE allGLOBALtokens. Prefillserialshared;decodeoverlap. Newmodelcompositionmatches.
StrictCPU8k1k:ADP2aggregate16complete;fouradditionaltensorlayouts eachagg+PD16complete, /tmp/aisim-tensor-layout-matrix-20261025.log.
New1296outputCPU/tmp/aisim-tensor-search-1296-20261025;notfrozen/committedyet. GoalstillACTIVE.

CRITICAL DCP AUDIT: all initial standaloneDCPplan dcp-attention-plan-20261025.json usedinterleave1,
butPINNEDNPUcall passesinterleave_size=backend.page_size128 atdeepseek_v2_attention_mla_npu.py509.
AscendBackend._get_kv_lens_and_block_tables also usespage128 andpage_stride128*dcp.
DO NOT PUBLISH ANYoldDCPstandaloneobservations (incl world2v3 hundredsamples/world16contract).
They remain diagnostics. Interruptedownedworld2v3;allown19containersstopped. layout-audit-rejection.json recordswhy.
Currentcorrectedplan production/dcp-attention-page128-plan-20261025.json still2736cases:912DCP2,912DCP4,608DCP8,304DCP16.
Currentcollector builds owned_positions=(global_position//128)%world, localindex=(pos//(128*world))*128+pos%128,
passes128toremapper. Allstate/datahelpersneedthissamecontract;neverfallbacktoCUDAdefault1.
Initialnumericalcase26 failurewasidealFP32referenceignoringBF16partialrounding. AddedindependentCPUchecks
forALLsource-head partials/LSE, thenCPUmergecheckedBF16partials. Outputtoleranceunchanged. IdealFP32errorretaineddiagnostic.
EmptyNPU partialLSE is finite-1.999999936e38, exp=0, notliteral-inf; directprobe dcp-empty-lse-probe-20261025.json.
ChecknegativeLSEandexp==0;finalreferenceignoreszero-weightpartials. Revisedoraclepassedcase26 andgraph-noopcontrol.
Thatoraclequalificationhadinterleave1, so nowrequalifycorrectpage128layoutbeforefullcollection.

ACTIVE19root89706: page128prefillprobe index26 thenpage128graphprobe index694 queued; outputs dcp-page128-{prefill,graph}-node19-20261025.
Neednegativepage128control thenfullworld2fresh;32freeforpage128world16positive/negative/full aftercontrols.
run_dcp_attention_campaign.py monitorsotherNPUprocesses andstopsoncontamination. Freshoutputs;collector onlywholeworld orcase-index.
CurrentDCPsourceuncommitted;doNOTformatwhileformalcollectionrunsshouldpinexactsourceSHA.
NoDCPnativeop/table/exporter/model/KVrewrite implementedyet. Planwholechainincludesremap(optional21/57),QAG,
sgl_sparse_flash_attention(returnLSE),FP32all_to_all_single+attention_update; excludesindexer/GEMMs/BMM/cachewrites.
ExistingHCCLdataBF16only, so donotpriceFP32mergewithit;wholeDCPchainwillmeasureactualcommunication.
DCPKVlatent+ropepage-striped,indexerFULLglobal. Capturedphysicalcap24832:K/V[78,195,128,1,512/64],index[78,390,128,1,128].
AdditionalCACHEAUDIT: DCPindexer scatter indices eagerINT32,graphINT64 (oppositeoldcache_scattercontract).
Latent/rope writes stillFULLqueryrowcount, notdividedDCP, withnonowned slotsdummy0. Needqualifiednewcachevector
roles/dtypes/ownershipdata;doNOTreuseoldindexerwritesunderwrongdtype. Oldreferencepoolleadingdimensionisexplicittimingreference.
Stillneed smallerTP1/2 if executable, DCPpredict/searchasfeasible,finalimage/docscommits. GPUReplaypp1guardconfirmed,
CPprefillruntimebroken,EP2noneisolatedglobal/localgroupsmismatch; donotremoveguards/fakeunsupportedcombinations.
Root93967localCPUfree;24914localfree;72844localfree;89706node19controls.31occupieduntouched.

## NEWEST: seven-layout525/525 stage validated; ADP2 tensor-MoE probe active

HEAD654d8936 source foundation committed. Newdefaultcatalog adds(16,1,16,1,1), sourceuncommitted withdata/docs release.
All525 auto candidates feasible,0failed; strictsourceprovenance. Frozenautomatic-search-525-20261025/recommendation.json.gz.
Tensor126,linear2183(old1378+805),vector1034(old842+192) INSTALLED; nativeexactqueriesallpasszeroerror.
Frozen5directories tensor-{moe,linear,vector}-release-20261025 andtensor-{linear,vector}-combined-release-20261025.
PublisherinitialbareimageDigestfixedtofullqualifiedsameimmutableimage; oldCPU-v1outputsretained, published-v2correct.
Moe-familymetadata v2 preserves per-table runtime:DeepEP ABI includesdeep_ep;newtensor runtime doesnotclaimit.
3CLIresults:8/8aggregated,16/16eachPDdirection;8k/1k token totals checked. ReversePDstrictprovenance passes.
390GPUgoldens passed469s while525sweep ranconcurrently; testwalltime isNOTmatchedFPEperformancecomparison.
47menu/model/provenance pass. Checkpointreleasepatchpending/applying revision57 withfrozenartifacts.
/tmp/aisim-tensor-validation-stage.json lists554selectedvalidationfiles;useexactlist,notrawrecommendation.json/logs.
NEXTcommitinstalleddata+catalog7+validation/docsaftercheckpoint. Userimage stillold190684; goalACTIVE.
IMPORTANTscopefinding:GPUrequestengine too rejectsPP!=1 inengine/launch.rs170;defaultGPUsearchPP1.
Do not build fake PP support byremoving thisguard. RealPP2DeepEP8works,butPPReplaynewfeaturewouldneedgeneralenginework.
CP2NPUmissingrebuild_cp_kv_cache proven;PP2none hasOwedOutput.shape runtimebug;EP2none isolatedglobal/localgroupsmismatchproven.
ACTIVE32 NPU container aisim-current-inputs-20261022-moetp16-adp2-20261025 port18000 loading;
TP16global,ADP2,EP1,none,KV32768,fraction.86,chunk4096. Need8krequest/captures andstopownserver.
19idle,31untouched. Root72844localfree,24914checkpoint,93967localfree,89706remote19free.
DCP2capture showsforward_dcp_sparse_attention ALSOusedprefill,notonlydecode. QAG78,partialattention78perphase.
NPUmerge cp_lse_ag_out_rs_mla_npu actuallyFP32 all_to_all_single +npu_attention_update despiteargag_rs.
ExistingHCCLonlyBF16;donotreuseforFP32. DCPKV:physicalcap24832,k/v[78,195,128,1,512/64],index[78,390,128,1,128].
Indexercache staysfull logical,latent+rope sharded;cannotdivideallKVbytesbyDCP.
DCPnewkernel/comms/packing+memorymodel stillneeded. SmallerTP1/2 mayneedsmallerprefillchunkdueDeepEPnormalM2048failure;
do notsimplyrelaxguards. Currentdefaultsevenlayoutsfunctional;remaininggoalstillactive.

## CURRENT: tensor-MoE data collected and native integration tested; publish tables next

Goal ACTIVE; nodes32+19 allowed,31 untouched. Checkpoint56 integrity[].
98381b91 committed six-layout351/351 auto search and transport extension.
Pinned replay baseline2434b021/fullSHA->98381b91:8rows32processes5000requests,all byte-identical;
frozen production/parallel-replay-validation-98381b91 (362 files). First short-SHA setup rejection preserved in /tmp.
Current native release d9fd5fa11ce791ba95c9ab70e70713983787d14df4a3d7d469b519c3b90f574d loaded atomically in CPU /tmp/aisim-native.
New AscendMoeTensor op/table/exporter supports EP1 balanced top8, intermediate64/128/256/512.
TP16ADP1MoETP16EP1 model composition now passes exact resident49,300,283,900 bytes and157AR16 trace checks.
36 Ascend native tests,14 producer tests,8 model tests pass; Ruff current source clean.
MoeTensor128 cases:126stable,2CVexcluded;4negativegraphcontrolsrejected. Failedfirstsmoke was checker token-major mapping bug,
proved by probe_tensor_routing.py; fixed only checker index, no tolerance or kernel changes. Frozen campaigns node19/v2.
Shared/dense new linear816cases complete805stable11CVexcluded;vector192complete192stable. Node19 own collectors stopped.
Production plans linear/vector-tensor-extension-plan-20261024.json are whole previously-unmeasured curves from complete declared plans.
Native linear/vector identities now percurve; new ascend_curve_append rejects overlapping curves, keeps all old data untouched.
Runtime identities moe-tensor-tensor-runtime-identity-20261024.json,linear/vector-tensor-runtime-identity-20261024.json.
NEXT:commit source foundation, export with exact collector-ref; merge linear805+old1378 andvector192+old842 using curve_append;
install126tensor table and preserved-family metadata; native-query all;full8k1k EP1 aggregate/PD;add default catalog when qualified.
Actual DCP2 fullmodel8k/16out passed withKV32768,fraction.86,chunk4096. Initial.9/8kchunkOOM retained.
DCP op/KV/LSE model remains UNIMPLEMENTED. All owned DCPservers stopped.
CP2interleave actual fullmodel failed warmup: missing DeepseekV2AttentionMLA.rebuild_cp_kv_cache atnpuMLAline490;
source only two calls/no definition. Preserve explicit runtime limitation; cannot claim generic CP support.
PP2DeepEP8 previously passed8krequest but PP model still UNIMPLEMENTED.
New PP2MoETP8EP1 failed graphcapture: deepseek_v2.py3169 accesses hidden_states.shape on OwedOutput while carrying topk_indices;
runtime bug at skip-indexer PPboundary. Need audit/other executable PP layouts; do not say all PP unsupported.
EP2none isolated actual AscendTPDispatcher probe failed:globalcounts256 versuslocalweight128 scale batch mismatch;
tensor-ep2-runtime-probe-20261024.json. This is isolated metadata execution evidence, not full distributed serving.
Archive /tmp/aisim-tensor-archive-stage.json selects completed newcampaigns/captures; never stage all untracked or md/test.md.
RootPTYS93967/24914/72844 local32,89706 node19. AllNPUscurrentlyidle after PP2failure; verify before next launch.
Old user image190684... remains previous limited release; must rebuild after current features qualify. Overall adaptation NOT complete.

## CURRENT ACTIVE: six TP/EP layouts and351searches pass; PP/MoETP execution proven, not modeled

Goal remains ACTIVE: userexplicitlyrequestsotherparallelwayslikeGPU;donotcloseafterthisstage.
HEAD1071459d;preceding5be38455(PDavg8),f45c5dfd(TP16PD),b712dd75(DeepEP256).
Checkpoint55 integrity[] saved;biguncommittedmodel/search/FIFO/PDcurve/data/docs patch needsfinishandcommit.
Sourceauto catalognowTP4/8/16 xEP16/32,ADP=EP/TP,MoETP1,PP1CP1DCP1.
351/35164-devicebudgetC8/16/32candidatespass. CPUoutput/tmp/aisim-parallel-search-351(notyetfrozenhost).
StrictprovenanceTP16P/EP32Dpredictpasses /tmp/aisim-pd-tp16-ep32-strict, frozenhostpd-tp16-ep32-strict-20261024.
Native loaded6c598dce2fa8bcda6ab4553516a1a845cc0f4bcaf9b99df8f56356167716916b,EngineSpec29.
NativePDloadernowallowsDIFFERENTcollectionidentitiesforDIFFERENTcurves,oneidentitywithineachcurve.
ascend_pd_merge.py selectsWHOLEcurvesfromcompatibleverifiedcampaignsby8kcoverage/span/count/earlierinput,
NEVERlatencyandNEVERsplicesdifferentprotocolswithinacurve;3Python+3Rusttests pass.
InstalledDeepEP143,attention1707,PD81(18curves). All143/1707/81exactnativequeriespass.
Attention1772numericcasesallpass,65CVexcluded;newgraphminimum16shapegridfixesTP16active1data miss.
validate_attention_native.py changedtouse[row.padded_batch]asthequeriedgraphbucket;oldfixedlistmisqueriednewshapes.
PDsingleTP16campaign90->71stable,avg8campaign90->66stable;eachaloneregressesfragmentedcoverage.
Combinedwholecurves81 retains8kcoverageforall18;sourcecampaignsformalold32,PD71,PD66allpreserved.
Frozenhostpd-tp16-single-release-20261024,pd-tp16-averaged-release-20261024,pd-parallel-release-20261024;
attention-tp16-release-20261024,deepep-parallel-release-20261024.
Fixedsdk/perf_database.pyv2collectionparsertoacceptclassified_failureswithnonnegativeintegercheck;
waswarningonnewmergedmetadata.Strictprovenanceand46focusedtests passafterfix.
Publicsuite984passed,5FAILEDonlyHFconnectionresetforQwen3-VLconfig (notcode);needresolvefixture/cacheorreportevidence.
GPU390paritypass116.17s. FullRust2042pass1preexistingignored BEFOREfinalPD-percurveidentitytestadded.
StillneedpinnedReplayparityagainst2434b021afterFIFOsemanticfix,fullcurrentchecks,andnewwheel/image.
ModelTP16resident51,292,091,900exactactualstartup;C1/17requestsrealweightsverified;TP16aggregatepredictnowpasses.
EP32TP8ADP4andTP4ADP8C4/8complete8k1k. PDperattn-group sourceproofallowsEP32withoutchangingpayload/rankmapping.
ActualPP2TP8EP8serveron32completed8k/16out;sourcecapturesoperator-input-state-node32-20261022-pp2-ep8-20261024.
ActualMoETP16EP1TP16servercompleted8k/16out;sourcecapturesoperator-input-state-node32-20261022-moetp16-ep1-20261024.
MoETP16capturesbothphases157AR16+1logitsAG16,noDeepEP. NeedsnewtensorMoEroute/compute/sharingmodel/data.
BothownedserversSTOPPED;allNPUs32/19idle.31occupieduntouched. Root24914/93967local;89706node19.
Argumentprobeinitialforgotresolve_once;discard/tmp/aisim-parallel-arguments-20261024asconstructor-only.
Correctprobe callsresolve_once;17casesaccepted,sourcefrozenproduction/parallel-arguments-resolved-20261024.
ThisisargumentvalidationONLY. DSA DCPcodeexplicitlyrejectsPDmode. CP/DCPkernelpathsremaintoexecute/model.
TemporarySGsources/tmp/aisim-sglang-parallel-audit-20261024;DeepEPsources/tmp/aisim-deepep-parallel-audit-20261024 (outsideRepo).
Archivehelper/tmp/aisim-seal-parallel-stage.pycompressed663selectedfiles+23logs;list/tmp/aisim-parallel-archive-stage.json.
DO NOTstagealluntracked:oldrawarchives,md/test.md,userfilesmustremainuntouched.
Sourceformatappliedusing/tmp/aisim-format-pd.py(v3dedup,skipbinary);lastRuffonlyPDmergestringE501needswrap.
MeasuredPDsourcea118f975...mustnotbeformattedagain;newobservercopiesremainasmeasured.
Nextcommitvalidated6-layoutstage/evidence;runpinnedReplayparity;thenMoE-TP,PP,CP/DCP/smallerTPextensions.

## Latest ACTIVE: 96/96 search fixed; TP16 needs padding data; PD averaging underway

HEADf45c5dfd(producerTP16PD)+b712dd75(DeepEP256protocol). Manyuncommittedsource/core/docs changesremain.
Checkpoint54currently;needsnextupdates. GoalACTIVEbroadparallelcoverage,notcomplete.
SGLangFIFOwaitingfixpassesred/greenregression,72schedulerand2042fullRusttests(1preexistingignored).
LoadedCPU/tmp/aisim-native/_runtime.abi3.so SHAa13d74ac6e9719bd98fa1a52109aa69e11566ab78374a5fc54b067e588558de6.
AutoCLIafterfix96/96feasible,noerrors; frozenhostproduction/automatic-search-96-20261024.
StillrequirespinnedlegacyreplayparityandGPUgatebeforefinalrelease.
TP16modelPythoncompositiondraft:TP16allowed,padding16,graph16/32,ADP1denseAR6,fusedpostdenseaddnorm.
Resident51,292,091,900matchesactualstartupallranks;memory-report.jsonverified16ranks.
TP16requestpredictioncurrentlyFAILSmissingattentionpadded16/active1brackets(oldgridroundedactive1to8).
Newattentionbasegraph_bucket_sets=[[8,16,24,32],[16,32]],getterdedupspaddedshapes;1772cases(+196),12producer tests pass.
Frozenproduction/attention-tp16-plan-20261024.json. NoTP1/2orMoETP/PP/CP/DCPimpl yet.
PDTP16first90casesallnumericallypassed,71stable19CVexcluded; /tmp/pd-parallel-release-f45c5dfd inCPUcontainer.
DoNOTinstallthat71tablealone:4to8fragmented8kregressesand8to16fragmentedhasmax4096. Original32formalrowsunchanged.
RevisedPDtimingaverages8calls/sample,warmup10,7repeats,keepsallper-calltimesandverifiesmeansinexporter.
7PDtests pass;graph/no-copycontrolrejectedonbothnodes. Sourceascend_pd.pySHAa118f975f7b8f2e73aa84c65ca56255fd933455f1f88ee0a40488181fe0f1cc4.
DO NOT FORMAT measured PDsourceagain:alreadyformattedbeforelaunch. Newplanpd-averaged-plan-20261024.json;
identitypd-averaged-runtime-identity-20261024.json. TheseandPDaveragingcodeuncommitted.
ACTIVE32root24914:run_pd_grid --suffixavg8-20261024 (/tmp/aisim-pd-avg8-node32.log).
QUEUEDON SAMEPTY:ifPDgridexit0,run_attention_campaign --run-nameattention-tp16-padding-node32-20261024 --planattention-tp16-plan...
log /tmp/aisim-attention-tp16-padding.log. DoNOTlaunchother32NPUworkwhilethispipelineactive.
ACTIVE19root89706:PDavg8peer(/tmp/aisim-pd-avg8-node19.log). Sparelocalroot93967.
FullDeepEP256published143rowsvalidatednativeall143delta0. Frozenhostproduction/deepep-parallel-release-20261024;
CPU/tmp/deepep-parallel-release-b712dd75. Formalold125NOTreplacedyet. Metadata mergeprepared
CPU/tmp/aisim-deepep-merged-meta.yaml preservesMoE-chain;needatomiccopytable/metadata/provenancewhenready.
RustPDKeynowallowsTP16andmodelguardallowsEP16/32withPP1CP1DCP1(uncommitted,NOTinloadeda13binary).
Sourceproof:CommonKVManager mappingusesengine_rank%localattnTP;PP1AscendcopyunaffectedbyADP/EP;
curveperremoteattentionDPgroupcanbereusedforEP32. UpdatedAPItestexpectsEP32samecurve,buildupdatenecessary.
FormalPD32/attention1517/DeepEP125tablesunchanged;runtimeimage190684...stilloldlimitedfunctionalversion.
Nextfinishcollections/publishwithpropernewsourcecommits;completeTP16/EP32defaultcatalog+PDguards/tests;
broadenTP1/2,MoETPtensorpath,PP,CP/DCPafteractualbackendproof. SGDSADCPexistsbutexplicitlyrejectsPDmode.

## Newest ACTIVE: automatic search exposed SGLang FIFO bug; TP16 state captured

Checkpoint54saved,resumeintegrity[];goalACTIVE. No newcommitsafter2434b021yet;uncommittedchangesmustbecompletedandstagedcarefully.
Auto-menu24unitstestspass. RealCLI96candidates=95passed/1failed:3P1D,PrefillTP4ADP4,DecodeTP8ADP2,C32.
Error:effect-freezero-durationpass22inflight;SGLangwaiting_for_external_command wronglyfalsewhenfresh/retracted
queueblockedbypendingdestinationFIFOandnorunning/prebuiltwork. Narrowcorefixchecksadmission_blocked.
Rustregressionredassertbeforefix,72SGLangtestsnowpass. DoesNOTremove/bypassReplaylivenessguard.
ACTIVEroot93967 releasebuild /tmp/aisim-pd-fifo-release.log;afterfinishatomicallyreplace/tmp/aisim-native/_runtime.abi3.so
andrerunfailedcandidate/all96;mayneedmorefixes. MustrunfullRust/engine/APIandpinnedreplayparityafterthissemanticfix.
Newtestinengine/scheduler/sglang/tests.rs;core.rs modified;bothchowned5031. Fixtureusesapply_command_effects(...,true).
Collectorfull256protocolcomplete:world32=40/44executed64stableboundaries;world16=40/44executed79stable;
143stableof176boundaryattempts,17CVexcluded,8failedcases=16missing. NormalM2048alwaysfails;4LLvector/GMerrorsrecorded.
Exporterpassedfull88plan: /tmp/deepep-parallel-expanded-published-20261024 inCPUcontainer;native143queriesrunningroot24914.
OLDformal125tableunchanged. DeepEPreaderrequireONEruntimeidentity:cannotmerge64/256protocolrows.
Runtimeidentityfileproduction/deepep-full-node19-20261024-v2/runtime-identity.json;collectorref434f8503/sourceunchanged.
ExternalDeepEPsourcesREADONLYinspectionmovedOUTOFREPOto/tmp/aisim-deepep-parallel-audit-20261024. DoNOTcommitexternalcode.
DefaultlowlatencypathisDefaultLowLatencyCommStrategy(deep_ep_cpp),NOTOps orAllToAll;DEEP_USE_MODEunset.
TP16ADP1realmodelcapturescompleted8kC1(16outputtokens)and128C17(32outputs),ownedserverSTOPPED.
Outputoperator-input-state-node32-20261022-tp16-adp1-20261024. server-info.jsonflatdictrecordsDP1,TP16,EP16,
enable_dp_lm_headFalse(normalizedfromTrue),graph16/32,max_total_num_tokens65664. KVpooltimingreference53kdiffersactualcapacity.
Bothprefillanddecodehave7AR16(embedding1+dense6),75RS16+75AG16sparse,1logitsAG16(lastdimension).
GenerationC1physical16,logical17physical32. LMheadGEMM[16,6144]x[6144,9680]atC1.
Densepostnormpath_reduce_update_read usesfusedadd+norm;oldDP2modelusesseparateadd+normanddecodeAG/RSdense.
NeedTP16composition:paddingmax(8,localTP),HCCLpadding//split(nonzero),graphbuckets16/32,
ADP1denseAR6andfusedaddnorm;actualmemoryinventoryverify;existingTP16compute-tableprimitivesmostlyalreadycollected.
MorePDreusefinding:CommonKVManager._resolve_rank_mapping usesengine_rank % localattnTP; Ascend.send_kvcache PP1
doesnotdependonADP/EP. EP32withlocalTP4/8canlikelyreusethesameper-attention-groupremoteKVcurveafterexplicit
scopeguardtests;doNOTjustrelaxwithoutrecordingthissourceproof. TP16PDstillneednewTP16mappingcollection.
AllNPUjobsnowstopped;32/19availableagain,31untouched. Root24914/93967local,89706node19.

## ACTIVE: broad parallel extension requested after limited EP16 delivery

User asks to adapt other parallel strategies like GPU AISimulate; existing ops should be reused wherevalid.
Newgoal ACTIVE, do not close after onlyautomaticEP16menu. Nodes32+19 authorized;31occupiedleavealone.
Resumecheckpoint53passed;newprogresspatch pending. SourceHEAD2434b021,uncommittedauto-menu+collector-protocol+launcherchanges.
ReadrepoAGENTS/perfmodel-api,FPEperformance,collectorlayer/failure/caseauthoring;noagentsauthorized/spawned.
ActualSGLangsourceaudit copiedreadonlyto /tmp/aisim-sglang-parallel-audit-20261024, commitbaa9befa.
Onlypyproject.toml/deletedpyproject_npu packagingdiffsinimage; srt sourcesunmodified.
GPUdefaultenumeratorPP1widthladder1..16;GPUDeepSeekV32supportsDCPonlyvLLM, BUTactualNPUruntime
HASDSADCP sparseLSEpath:hardware_backend/npu/attention/dsa_dcp.py andmodules/deepseek_v2_attention_mla_npu.py.
DeepEPoverridesEP=globalTP. MoETPneedsdifferentactualAscendnone/ascend_tppath. PPneedsindexersharingboundaryaudit.
Draftmodelhelperseparatescheckpointvalidationanddefaultshape tuples;ModelHardwaregetsdeclaredshape menu;
genericreplica/pairhelpersallowallbudgetedratios.Pytests24pass /tmp/aisim-parallel-menu-tests-v2.log.
ACTIVECPUroot93967:96candidate autoCLIrun /tmp/aisim-auto-parallel-search.log;outputcontainer/tmp/aisim-auto-parallel-search.
PYTHONPATHmustbegin/tmp/aisim-nativeforupdatedsourceimports;installedwheelalonecannotattachnativewhenpackageSRCoverridden.
ACTIVE32NPU:aisim-current-inputs-20261022-tp16-adp1-20261024,port18000,weightsloading/stateobserver;
root24914free.Launcherhasnew--attention-dp1 --decode-graph-buckets16 32,allcaptureflags.
Outputoperator-input-state-node32-20261022-tp16-adp1-20261024;requestshavenotbeendrivenyet;muststopownedserverafter.
ACTIVENODE19root89706:run_deepep_campaign.py local16world16 plan256calls/warmup10/repeats7;
outputdeepep-parallel-extension-local16-node19-20261024.
Dualnode32/19v2finishedall44:40passedphysical,normalM2048cases51/59tilingfailed,
LLcases68(eagerbalancedM8)and87(graphhotM32)NPUvector/GMerrorrecorded.Original40validpreserved.
BalancedgraphM1/M3/M32bothboundariesCVpass=>smallbatchbounds;M2/4/8combineCVfail,M16bothCVfail.
Fullplan88mustcompleteworld16beforeexport;formalold125rowstableNOTREPLACED.
Newplanproduction/deepep-parallel-extension-plan-20261024.json;sourcecollectorunchanged.
Initialnon-v2directoriesonlysetupfailure(missingplanbecausehostPythonlacksPyYAML), noNPUwork;preservelogs.
NextfinishautoCLI/format/docs/checkpointandstagecommit;completeandpublishnewtransportdata;
observeTP16realboundary/padding (existingcollective8//TPwouldbezeroatTP16;donotjustrelaxguard),
thenexpandTP/MoETP/PP/CP/DCP withsource-qualifiedcollectorsandnativecomposition.

## Delivered: functional EP16 operator prediction and search

Finalimage sha256:190684496125b32f2cc58531cae4676a4e229674c37bbd5e8caff97802fa5a2e.
scripts/run-ascend910c.sh uses installedunifiedwheel,networknone,non-rootUID,CPU16,memory32GiB,noNPU.
Actualmodel/mnt/raid/user/data/models/glm-q:PD16/16requests,unified60/60candidates,best64/64requests,
bayesian12suggestions=11successfulreplays+1cachehit. All60metricsidenticaltostagedmetadatabaseruns.
Installedpackageverificationand395GPU/Ascendtests pass.8legacyreplayrows/32processes/5000requestsbyteparitypass.
Summary production/functional-delivery-20261024/summary.json andmanifest;usage md/910c-glm52-w8a8-usage.md.
Publishednative/sourcef90015df; runtime/docs/evidence form the final delivery stage recorded in git log.
NoNPUjobsactive;31untouched. Functionalgoalreadyforhandoff withinqualifiedEP16domain.
Remainingfuturework:servingaccuracy,imbalance/realrouting,PDcontention,EP32smalldecode/newtopologies.
Do not claim formal direct-FPM profile onboarding or unrestricted hardware/model topology support.

## Final delivery in progress: wheel and replay regression complete

Committed functional stage f90015df. Unified search60/60 and PD48/48 pass; bestYAMLs replayed.
Replay baselinea93487f9/candidatef90015df:8qualified5000rowMooncake configurations,2runs/revision,
all32canonical reportsbyteidentical. Requalified legacy seeds onbaseline;SGpressure3/1/2,VLLM2/2/3,
allpressure readmitted,allrequestsfulloutput,reuse/queue/handoffverified. Nativebenchmarkconfigsunchangedbetweenrevisions.
Artifacts replay-validation-20261024, pd-replay-parity-20261024.json, unified-search-20261024,
unified-best-replay-20261024. Skillrunnerlockrefreshedidentically(olderlockfailedboth);exactlocksarchived.
Distributionwheel built withmaturin1.15.0,sourcef90015df,EngineSpec29;
nativeSHAaa5cec4945b59d306f85dcf78d7b341a617224f7a39de6f6a15a6b5e6edbb438,
wheelSHAd3f58f0bedfecc57c6cb7764e35ad53dfe3784030911651845b8fa37d10bc1ec.
Installedwheelnativepassed395GPU/Ascendtests68.50s. Frozenpre-wheelbinary44358...alsoexactrebuildf900.
Runtimecode usesinstalledunifiedwheel(noPYTHONPATHnative shim). CurrentCPUimagebuild root24914:
/tmp/aisim-runtime-wheel-image-build-v3.log. All33dependencywheelsverifiedPyPIsha andcachedin
ignored build/ascend910c-runtime;oldslowonlinebuildcontainersa008d2bab8e5,5e69ace6a8festopped(ours).
FinalDockerfileinstallsfullrequirementsfromlocalwheelhouse;addsCPUcpuset16,quotalimit16,memory32GiB.
New scripts/run-ascend910c.sh and build-ascend910c-runtime.sh,md/README.md andusage.md drafted.
Remaining:runwrapperwithactual/mnt/raid/user/data/models/glm-q(PDpredict,unifiedrecommend,bestYAML,
bayesian smoke),recordimageidentity/pipversions/finalmanifest;updatecheckpoint52;finalstagecommit.
Initialruntimeimage900e6283taghadbadnativeimportlayout;doNOTuseuntilv3buildreplacesit.
NoNPUjobsactive,31untouched. Spare root93967. Usergoalstillunfinisheduntilruntimedeliverytested.

## Newest: measured PD and 48-candidate recommendation pass

PD producer committed a93487f9. Native/table/replay integration ready for commit.
Native SHA44358dc3d1e56c7cbe56c3484c98bec27b22d459867dcc77535eafe9f756e6a8, EngineSpec29.
32/40 stable PD rows installed kv_transfer/sglang/0.5.22, 8 CV exclusions preserved.
Canonical role model API selects page layout and TP mapping; rejects non-16-device-per-role layouts.
32 exact queries pass; C1 full8k/1k replay source hold duration7.781459979ms agrees with curve.
C4/16request PD replay completes. C8/16/32, role TP4/8 ADP4/2 EP16, P/D replicas1/2:
48/48 recommendations feasible, no failures; best saved YAML replayed successfully.
Prefill chunk4096 avoids unqualified TP4 per-rank2048 normal DeepEP; decode max_sequences32.
2041 Rust tests, 916 public config/runner/search tests, 9newfeature checks,390GPUparity117.37s pass.
Logs archived pd-validation-20261024; PD raw logs compressed with archives.json in16campaigns.
No NPU jobs active. Still need durable CPU runtime/user commands and pinned replay byte-parity.
All source/math uses canonical Rust model; no Python estimator or GPU donor substitution.
The broad goal is not complete until these final user-facing functional checks/docs are done.
RootPTY24914/93967 local;89706 node19. Node31 remains unavailable. Checkpoint51 pending update.

## Current: complete model/engine prediction works; PD grid complete, consumer pending

Canonical best_available(op_level) and predict --stack engine pass for8k/1k,8requests,concurrency2.
Example examples/cli/ascend910c-glm52-8k1k.yaml usesrealhostmodelpath;validationoverrideswithidenticalmetadata snapshot.
Reportengine-canary-8k1k-20261024/prediction.json;TTFT2642.412883ms,TPOT108.555693ms,18.013123outputtok/s.
Thesearepredictions,notservingvalidation. UniformADP andbalancedexpertprofileexplicit.
Native239d22cd3a8b4e5f89e7faa2ae2cae88b1de38538fe2bfe938216e2d1103f36d,schema29.
Modelweights52,471,997,948bytesexactmatch. Newresident_copiesfieldsforvector/BMMandHCCLrequesttokenaxis.
29Ascend+6HCCL+22wire+82Python/API+390GPUparitypass116.34s. Lateststrict/mixedsmokelog/tmp/aisim-forward-strict-mixed.log.
YAMLmisc.other_mem937753092measuredmaxof32snapshots;nccl_mem0actualNCCLabsent,notHCCLproxy;SDK20%margin.
Checkpointrebaselined50,latestmodel-functional51. Detaileddatareportsmovedundermetadata/tofixstrictadmission.
Logitscollector sourcecommitbcc065a7,848cases842stable6excluded,18noopcontrols;frozenvector-logits-release-20261024.
Actualboundarycapturecomplete,sourceandrawarchived;model-statecards16. No standaloneonlinequarotwork.
PD actualmemfabric1.2.1SDMAprobespass128and8192tokens(fullpayload899678208bytes=7.756787ms),nooprejected.
Full32processsmokepasses. FormalPD40casesallcompleted:pd-probe-npu1-32-pd{88,48,84,44}-20261024,
peerlogsnode19.32stable8CVexcluded. All8kpointsstable:88contig7.7815ms/fragmented66.332ms;
48contig15.5202ms/frag130.4192ms;84contig7.7453ms/frag65.3545ms;44contig7.7941ms/frag65.1291ms.
Newcollector asc_pd.py/cases/exporterdrafted;exporterNOTtested/publishedyet. Planpd-plan-20261024.json.
NOACTIVE NPUjobs. Root32=24914,93967;19=89706;31leavealone. GoalNOTcomplete.
NextimplementPDtable/nativequeryviareturnedcanonicalmodel;wirepublicengine.kv_transferproviderandhandoffcurve;
thenrecommendparallel/P:DsearchanddurableuserCLI. ExistingKVtransferonlybandwidthanddefaults0delay.
Suggestedintegration:ReplayEngineConfignewproviderflag,resolve_role_timingoptionallyreturnscanonicalmodels,
derivecurvefromPmodel+Dmodel(oneconstructor),ReplayEngineFactoryholdsArcCurve,DisaggRuntimepasses
itintoperrequestHandoffCoordinatorCore;HandoffTransferTimingcanstayCopy. Core.advance_activeevaluates
curveagainstfull/destination-missingtokens. KeepdefaultGPUbehaviorunchanged.
Do notspendmoretimeonaccuracytuning;currentdata-domainmissesstayexplicit.

## Latest: DeepEP installed; actual United layer boundaries observed

Compute data commit163001f2;DeepEP/codecommit434f8503. NewDeepEP125stable rowsinstalledmoe/sglang/0.5.22,
separateascend_deepep_provenance.jsonandmergedcollection_meta preserveMoEchainmetadata.
All125nativequeriespass. Frozenproduction/deepep64-published-20261024 includescoverage.json.
All88configsattempted:84valid=168boundaries,125stable43CVexcluded;4normalM2048tilingfailures=8missing.
World16balancednormalM1..1024andgraphLLM1..32covered. World32balancedgraphLLonlyM16stable;
do notpretendEP32smalldecodeiscovered. Stopaccuracytuning;focusfunctionalcomposition.
Finalcampaignslocaldeepep-full-node19-20261024-v3,dualnode32/node19-20261024-v2;plan-v2(64calls/sample).
Alltransportjobsstopped. Archivescreatedforfinishedtransportdirs;notyetcommitted.
Boundarycaptureoperator-input-state-node32-20261022-boundaries-20261024 requestscompleted,ownedserverstopped.
SourcechangesInputCapture.describe+AISIM_CAPTURE_BOUNDARIES/launcherflaguncommitted,lint/formatpass.
Actualprefill8kglobal[8192,0]:embeddingAR8x1,globalAR16x6,attnRS8x75,AG8x75,logitsAG8x1.
Graphbatch1physical8:dense16,shared1;embeddingAR8x1,attnRS8x78,AG8x78,denseAG16x3,RS16x3,logitsAG8x1.
ModelSDKbatchisperattentionDP;balancedprefilltotalDP2 doublesdenseglobaltokens. TokenScale scalesbothbatchandtokens.
StageBoundarypaths:inputnormlayer0..3onattnM;layers4..77normonM/8thenAG8.
Densepostattn_dp_gather_sum_read normonglobalM;MoEpostattn_reduce_scatter_update_read normonM/8.
Needfinalnormsource/shapeandaggregateallranks;rawStagePathserializedactualfunctions/inputs available.
NoonlinequarotGEMMobserved;actualstartup saysrot.weightnotfound;indexerparameterlayers21only.
PDsourceAscendTransferEngine(memfabric-hybrid1.2.1)defaultSDMA,notHCCL. ActualoldPDscriptusesascend.
AscendKVManager.send_kvcache PP1 batches234buffers(78x3BF16components),groupcontiguouspageindices.
Needinspectactualsender/rankmappingandcollectPDtransport;notdone. NoGLMfullmodelcompositioncodewrittenyet.
CanonicalOpLevelConfig emptyinperfmodel/fpm/estimator.rs. Anynewroutingpolicycontrols belongthere,
notPython-onlydefaults. CurrentlegacyModelConfig.workload_distribution=power_law;Ascendtablesaresyntheticbalanced/hot.
Rootsessions32=24914,93967;19=89706. Latestcheckpoint48beforependingpatch. GoalNOTcomplete.

## Latest 2026-10-24: TP32 compute published, DeepEP first complete grid attempted

Source434f8503 adds DeepEP native Op52 (53variants,schema28) and globalTP32 shapes.
Installed linear1378rows/1382attempts,vector792/800. Exact native2170queries pass;
3newRust+22wire+15Python/API+390GPUparitypassed111.01s. Loadednative36b3fc6f1dfd813f94ec953213cf64c77436d84fde4ade0b5df42c436a5b3eaf.
FirstDeepEPlocal16anddual32/19 complete:88configs,84valid executions,4normalM2048tiling failures.
83stable timingrows outof168executedboundaries;85CVexcluded;previewonlyinCPUcontainer/tmp/deepep-published-final-20261024.
Essentialdecodegraphcoverage missing,so extended averaging8→64 inbaseYAML,plan-v2.json.
ACTIVE:19deepep-full-node19-20261024-v3,port30020+attempt. Inspectcampaign/observationsbeforeanythingelse.
Then runnewdual32/19 withsameplan-v2/source,coordinatedrunner. Olddualcompleteowncontainersstopped.
Producerretainsgraphs,onecall/capture,per-callouterevents+unmeasuredbarriers;bothnoopcontrolsrejected.
Initialwarmuphangandallocatorerrorpreservedinoldercampaigns. Failuresaredata;runnerrestartsaftereachrecordedfailedcase.
Exporteracceptscompleteplanacrosspartialattempts,neverfabricatesfailedrows;CVgateunchanged10%.
RootPTY32=24914and93967;19=89706. User31occupied,leaveuntouched. Checkpoint47beforelatestpatch.
Modelcompositionnotstarted:SDKbatchisper-attention-DP(base_backend.py1146global_bs=bs*ADP).
NativeTokenScale scalesbothbatchandtokens. ActualModelSlimindexerparametersonly21layers[0,1,2,6,...74],
FLOATkv_b,staticW8A8q/kv_a/q_b/o,indexerwq_b;dynamicMLPs;FLOATindexerwk/weights/router/lmhead/embedding.
Rotationdeclaredoptional/quarot.safetensors;sourceauditneededbeforecomposing(unconfirmedruntimework).
Memoryreachablemodelstorage~52.472GB/rank16;rawreportdoesnotyetclaimproductionmemorymodelready.
FullGLMcomposition,memory/PD andcanonicalpredict/recommend stillrequired;goalNOTcomplete.

## Newest: native compute data published; DeepEP contract probe on19

Sourcecommits589804d8(MoEchain+multi-requestprefillnative),5621929e(attentionproducer/failurehandling).
Loadednative3c2a659b421b8e02d61e0b6dde8e9272a17ddb5f6de047f7516d8d87210cc2b7.
OpindicesAscendLinear47,Attention48,Bmm49,Vector50,MoeChain51;total52,schema28unchanged.
MoE217stablephysical/220attempts→225profilerows (8aliasesone-token). Publishedstandardmoe,
frozenmoe-chain-published-20261024;225nativequerieszeroerror,4Rust22wire19Python/API390GPUparitypass114.84s.
Attentionnew1576gridcompleteattempted1517stable58CVexcluded1numericfailure;publishedstandardattention,
frozenattention-expanded-published-20261024;1517nativequerieszeroerror11collectortests.
Failure1137 deterministiconeelementactual0versusCPU-.0673 evenactualeager+rawBF16copy+FP64reference.
Readcollector/failure_handling.md:isolatedrecordedfailuredoesnotinvalidatedataset. Addedper-case
unexpectednumericalfailure records,no timingrow,no weakenedthreshold;systematicfailuresabort.
No moreaccuracytuning/debugloopneeded;failurefullyrecorded. Originalprobesinlinear-tiling-node32-20261022.
HCCLnow449stable,95excluded,frozenlarge-localpublished;installedYAMLunchangedselectsnew449table.
Allmajorcomputeprimitivespublished;fullGLMcomposition,DeepEP/memory/PD/searchNOTdone.
DeepEPprobeon19 owncontaineraisim-deepep-contract-20261024 (rank16hostPID1851613),
production/deepep-contract-node19-20261024;hostlog/tmp/aisim-deepep-contract-20261024.log.
UsesactualSGLang_ImplNormal(async_finish=True)and_ImplLowLatency(return_recv_hook=True),
DeepEPMode.AUTO,dispatcher_output_dtypeint8. Sourcepinnedframeworkbaa9befa,DeepEP1.0.0+0234a271.
Defaultstrategiesselected,DEEP_USE_MODEunset. Probenotperftiming;normalidentityroundtripmayneed
Ascendrunnerpermutation/weighting. Inspectresult/logbeforelaunchingmore19NPUwork.
32NPUsidle;31occupied. Rootsessions32=24914,19=89706,31=47855. Checkpointlast45beforelatestpatch.

## Newest 2026-10-24: MoE collected, attention extension has numerical failure

User says finishcurrentadaptation;19reauthorized. Currentcommittedvector/evidencestage
aded309b;sourcebd6f41f5,observer75584894;HCCLselector10b7a5ff. Checkpoint44beforelatestpatch.
MoEchaincollectedon19:moe-chain-full-node19-20261023-attempt2 completed220cases217stable,
8negativecontrolsrejected.220physicalfrom228recipes;balanced/single_expert syntheticprofiles,
8weightcopies. Firstattemptemptyfiltererrorretained. Own19containerstopped.
NativeMoEtable/opdraftappendedindex51,total52(notbuilt/tested);exporter/testsnotyetwritten.
Attentionmulti-requestprefill/nativechangesdraft+basegrid1576. Three394shardscomplete,
shard1failsindex1137after284. Existing505publishedtableNOTreplaced.
Failcase1137:decodegraphheads8M8active8KV8192pool513. Oneoutputelementqueryrow4zero,
CPUreference-0.0673,tolerance.0208;allothersclose. Diagnosticv2comparesactualeagerkernel.
Inspect/tmp/aisim-attention-case1137-diagnostic-v2.log (root24914) andfrozenprobeoutputs
inproduction/linear-tiling-node32-20261022 beforeadjustments. Do notwaivenumericchecks.
Nativeattention5Rusttests passed;Pythonoldfixtureselectorfixedactive1,rerunneeded.
Releaselibrary/tmp/aisim-target/releasebuiltwithattentionmulti;loaded/tmp/aisim-native
stillvector539802dc. NativeMoEnewdraftnotbuilt. Frozenvectorreportexact765passed.
HCCLlarge64rowsmerged:installed449stable95excluded,sameYAMLselector;frozen
hccl-large-local-node32-20261023/published. Native449validationoutput/tmp/aisim-hccl-large-native.json.
Needarchive+committhisdatastage andfinishMoEexport/native,DeepEP,model/memory/PD/search.
32rootPTY24914;19rootPTY89706;31rootPTY47855otherworkload. NPUjobonly32diagnosticv2.

## Latest: vector complete; switch future peer to19

User: firstfunctionality,accuracydeferred;now31occupied,19reauthorized,finishcurrentadaptation.
19rootPTY89706:16NPUsidle,driver26.1.1 nowmatches32. Existingxqcontaineronlysleep/bash,
do nottouchit. Model/mnt/raidpathabsent;othercontainerhasRO/data_lib/data/models/glm-q,
notyetfingerprinted.31rootPTY47855othermodeloccupies16chips;firstDockerstart14:43:48CST,
afterourHCCLfinished14:06,sooldcurrent31/32dataclean.32rootPTY24914.
Vector sourcebd6f41f5,observer75584894.768cases765stable3excluded;17negativecontrols
rejected;765nativequeriesexact;4Rust22wire17Python/API390GPUparitypass112.32s.
Native539802dc5399caa9ca9519717b3f151bbb2c92fea5f2242e9efa4fba31d91c85 inCPUcontainer.
Frozenvector-published-20261023 andinstalledvector/sglang/0.5.22.
Actualvectorstate107requests288rankforwards,graph8/16/24/32;16rankmemoryvalidatedKV53248.
Source/evidenceoperator-input-state-node32-20261022-vectors-20261023-attempt2;
firstattempthealthsocket.timeoutPython3.9failurepreserved,C4firsthealthrecordexcluded.
Observergraphmemoryincludescountcopies. Directnormbiascalls werenothooked;sourceaudited.
AllownNPUtasksstopped. Newhccl-large-local-node32-20261023 completed64stablecases
192/384MiB2/4/8/16ranks;mergewith385currentlyinstalledusingnewprovenancekeywithout
overwritingfrozenregisteredtable. No PDhandoffqualificationyet.
NextpriorityMoE/DeepEP actualeager+graphchecks,thenactualmodelcomposition/memory/PD/search.
Attentionstillneedsbatchbuckets9,17,25endpoints,prefillM8,eager32-widecasesforanyfullsearch.

## Latest after resumption: current-node HCCL published and selected

HCCL31/32 campaigns bothcompleted,taskcontainersstopped.160newcases allnumericalchecks
pass;107stable,53CVexcluded. Combinedwith278oldlocal32rows(samesource/image/driver/protocol)
gives385stable/480attempted. No node19relabeling. Frozenhccl-node31-node32-20261023/published.
Installedcomm/hccl/9.1.0-b243-drv26.1.1-nodes31-32-v3;ascend910c.yaml nowselects it.
385nativequeriesusinginstalledYAMLpass;390GPUparitypass117.54s. Runtimebinary stilld60599aa.
Hardwareinputchangedonlymisc.hccl_version;oldYAMLsnapshot preserved. Checkpoint42rebaselined
source identity,archivedhistorical collection/validationrefs,andregisteredcurrent4tables/provenance.
Formalprofile remainsunaccepted/stage2. Fullmodelcomposition/PD/search stillincomplete.
GoalUIwaspausedbutlatestuserexplicitlysaidcontinue;continueauthorizedwork(toolcannotresumeit).
Nextvector/norm/RoPE/cache andMoE/DeepEP. Node32rootPTY24914;31rootPTY47855.

## Latest: BMM complete; current-node HCCL running

BMM source30ab4198,222cases/220stable/2CVexcluded. Allactualexecutionchecks pass,
3graphnoopcontrolsrejected. Attempt1filenamefailureandattempt2oversizeddecode
batchIdxfailurespreserved. Formaldecodeaxis1..32matchesUnitedperDPrequestlimit.
Publishedmla_bmm/sglang/0.5.22 andfrozenbmm-published-20261023.
220nativeexactquerieszeroerror;4Rust+22wire+5collector+11API+390GPUparitypass.
CurrentnativeSHAd60599aaccf0a0b9823e719a624d9cd5f115ef7d68ed3fad1a39c8538dfc1444.
Node32rootPTY24914;node31SSHrootPTY47855. Nodesbothidlebeforelaunch;31driver26.1.1.
run_hccl_current_nodes.py active onbothnodes inhccl-node31-node32-20261023.
Node32hostPID3169765,31PID2447073;owncontaineraisim-hccl-current-20261023.
Collects2node groupswith8and16rankspernode (world16/32),sourcev3unchanged.
Inspectsharedpernodecampaign.json andrawhccl-16/32.json beforeanyrestart.
No servinginstancesstarted/stopped;originalservicesremainasfoundafterreboot.
Nextsourceauditvector/norm/RoPE/cache,thenMoE/DeepEP andactualmodelcomposition.
No complete canonical GLM forward/PD/recommend pipeline yet;goal remainsactive.

## 2026-10-23 attention collection and native integration

Source commit0a6ac76e.528/528attention/indexer cases completed;505stablepublished,
23CVexcluded;all before/after numerical checks and both graph-noop negative controls passed.
Frozen attention-published-20261023, installed attention/sglang/0.5.22.
505nativeexactquerieszeroerror;5Rust+22wire+9collector+11API+390GPUparitypass.
NativeSHA44fe026f8b862675e2969cb61eb247c0e2d7da5f41215789728d866a67008587.
Attention query distinguishes active batch, capture padding and KV length. Prefill
currently single request; graph decode intermediate batches need same-bucket brackets.
No accuracy calibration claim. Full model composition/PD/search remains incomplete.
Next collect actual MLA BMM paths: query input slice stride[heads*256,256,1],
w_kc stride[98304,1,192]; prefill output torch_npu transpose BMM, decode output
custom torch.ops.npu.batch_matmul_transpose with explicit destination.
Current node32 rootPTY24914; CPUcontainer alive, attentioncollector stopped.

UPDATE AFTER ABOVE-SUMMARY WORK: checkpointrevision38. Native mixedsourcecommitted
c774dc37;inputobservercommittedd501ecf1. New1241stablemixedrowsfrom1246casesinstalled
standardascend910c/gemm/sglang/0.5.22 andfrozenlinear-mixed-published-20261022.
Native1241querieszeroerror;7Rust,14collector,13versiontestsand390GPUparityPASS.
Ascendquery_versions.yaml overridepins0.5.22soNPUdata cannotpromoteCUDAfleetnext
(firstparityrunhad6version-gatefailures;fixedrootcause,neverchangedgoldens).
CurrentRELEASEnativeSHA4f3e65299344a4477dba034c0e32cc2e743a1d2e580b6e329ba57ea9391f5ad5.
AllNPUcollectionjobsended. Own31recordingattempt2STOPPEDafter11requests;noactiveNPUjobs.
CPUvalidationandlocalidleaisim-linear-tiling-20261022available.
Data/evidencestagecommitstillneeded;manynewartifactfilesuntracked. Preserveoldmd/test.md
andoldPID/lock/logleftovers. Inputplainjsonlarchivedtogzipwithinput-archives.json;
originalplainfilesignoredlocally. Donotmodifyregisteredimmutableinputreport.
Immediatepriority: commonnativeNPUprimitivecontract forBMM/vector/attention/indexer,
source-correctcollection,actualGLMcomposition;thenMoE/DeepEP,memory,PD/search.
DoNOTresumeaccuracytuningoradditionalheldoutcampaigns(userexplicitlydeferredthem).

LATEST USER PRIORITY: complete collection/adaptation and makepredict/search/PDusableFIRST.
DEFERpredictionaccuracytuning. Preservebasicnumerical/source/shape/precisionchecks;
failed10%heldoutgateisnotafunctionalbringupblocker. No moredensity/heldoutcampaigns now.
Checkpointrevision37(afterfunctionalpriorityupdate),integrity[].

Latestverifiedinputcampaign:
operator-input-state-node31-20261022-attempt2/report.json:96rankforwards,11complete
8192/1024requests(C1earlyandC10late). C1prefillattn8192,dense8192,shared/router1024;
graphC1andC10attn8,dense16,shared/router1. PaddingKVlength0. Fullindexer21/sparse78.
IndexerweightsprojectionactuallyFP32input/BF16weight/FP32outputthenBF16cast!
Mixedfloat32_bfloat16variantaddedtoAscendLinearkey/collector/exporter/cases;
7Rusttestsand14Pythontests pass,mixedM8+M16384/64copysmokeandnoopnegativepass.
Pendingfullmixed1246gridon32:linear-mixed-full-20261022,hostPID2745420;
releasebuild /tmp/aisim-mixed-release.log alsoactive. RebuildinstallandGPUparity
neededafternewmixednativechanges;oldreleasecurrentlyinCPUcontaineris5aac6ea3.

Newoptionalascend_input_capture.py+ascend_recording_server.py changesuncommitted.
Initial31capturefailedcustomtorch.opswrapperlost.default. FixedObservedCallable
proxyforattributes,4CPUtestsandrealcustomNPUgraphchecks2pass. Fullcapturepassed.
LMheadnotcapturedbyF.linear(ituseslogits_processor torch.matmul),sourceauditpending.
Observerstate/evidenceonly;notforwardtiming.

Currentmodel32fullidentity41795173ae457a64056e1c05be9bbbf889f935431f4a42fa4f4e4c6a32aa931c.
31differs13files;verified13overridesread-onlymountedforfullservingcapture.
model-view-plan.json/completedmodel-overrides-copy.json provecomposed196filesmatch.
Originaldirectoriesnotmodified. Own31captureattempt2nowSTOPPED;earlierfailedkept.
AllNPUjobsendedexcept32mixedfull. CPUvalidation+localtilingcalibrationcontainersalive.

Initiallinear657tableinstalledatstandardpathandcommittedfcbbfc9c. Denser1242candidate
underCPUcontainer/tmp/aisim-linear-dense-published NOTinstalled(maxheldouterror43%).
ProfileridentifiedMatMulV2blockcountjumps(3072->3328 M gives24->13blocks),butstop
accuracyinvestigationperuser. Preservelinear-dense* andlinear-tiling*evidencewhencommitting.

Full explicit goal remains ACTIVE: actual operator-level Ascend910C/GLM5.2
ModelSlim W8A8 prediction, resource accounting, PD handoff and canonical
predict/recommend searches over supported parallelism and P/D replicas.
Do not mark complete from foundation/data-capture tests. No blocked condition.

User currently authorizes nodes31and32; do not use19. Coordinated collector/consumer work, own-service
pauses, collection and local commits after completed stages. No branch/push/PR.
Git as youxingye; command author youxingye <youxingye@npu1-32>. Use runuser only fromroot.
Local UID5031. Never persist credentials. Preserve md/test.md,
checkpoint lock and old PID files. Root-created task files need chown to5031.

## State now

- Latest commits:
  - 5aac6ea3: AscendLinear native operator/table, explicit ModelSlim collector, exporter and provenance.
  - 76454223: observed-count MoE compute and cache-protocol experiments.
  - f5349d2f: Ascend KV layout component and startup/serving observations.
  - dd1eb7ab: fix actual HCCL graph correctness, recollect and qualify v3.
- Uncommitted linear evidence/data publication and heldout validation remain.
  Source primitive is committed; canonical whole-GLM composition still missing.
- Nodes31/32 rebooted on2026-10-22 around12:20. All prior Docker services were
  already stopped when resumed. Do not claimUnitedhealthy. Both modelRAID5 arrays
  are degraded4/3 and were unmounted; mounted read-only,norecovery for modelreads.
  No filesystem repair, rebuild or write mount performed. Other workloads untouched.
- CPUvalidationcontainer restarted; image/native/cargo caches survived initslayer.
  Host/tmp campaign/deps vanished. MinimalCLI ijson restored viaCPUcontainer;
  single-node/checkpoint.py worksagain. Its failures hideCLI stdout; inspect
  CalledProcessError.stdout whenneeded. Root-created checkpoint needschown5031.
- Checkpoint revision33, integrity issues[]:
  ../onboarding-checkpoint.json. Resume CLI FIRST on continuation with
  single-node/checkpoint.py --log /tmp/<fresh>.json.
  Current configuration united-tp16-dp2-ep16 remains stage2/in_progress.
  Formal exact FPM profile is not accepted; onboard init is vLLM-only.
  Operator adaptation can continue. Do not invent accepted direct-FPM profiles.
- Current goal is incomplete. Next priority is canonical Ascend operator
  composition and required compute curves, not more state-only collection.

## Linear phase just completed

- Full660physicalcases from768recipes,64weightcopies,128calls,10warmupbatches,
  11repeats. Allnumericalchecks pass;657CVqualified,3unstableeagerpoints retained
  (indices0,172,384). EagerQKV_A M1isabsent; no extrapolationbelowM2.
- Exactnative657queriesgivezeroerror.6Rusttests,22wiretests,60collector/provenance
  tests,390GPUparitytests(110.31s),focusedRuffcheck/formatpass.
- Dataset installed(generated, notyetdatacommitted)under
  python/aisimulate/src/aisimulate_core/systems/data/ascend910c/gemm/sglang/0.5.22/.
  Exportercollector_ref=5aac6ea3;metadataclosure/sourcehashes retained.
- M8paired31/32comparison40cases max4.3964%;32/64copy sensitivitymax9.3743%.
  Thisisboundedobservedsensitivity,notfullcacheconvergence/wholemodelaccuracy.
  Sources:linear-controls-analysis.json,linear-cache64-node31/node32-20261022.
- Native RELEASE /tmp/aisim-native/_runtime.abi3.so insideCPUcontainer nowincludes
  AscendLinear;SHA676dac63c5d87a512f7ce0b60fa566b209d356a0f5183de0fe2af33136becb12.
  Oldbinarybackup /tmp/aisim-native/_runtime-before-linear.so.
- Completed heldout188casecampaign on31 useslinear-holdout-inputs-20261022/plan.json,
  outputlinear-holdout-node31-20261022;containeraisim-linear-holdout-20261022.
  PlanM3/12/48/96/3072,headrequests3/12/48. Validatewithvalidate_linear_native.py
  --holdout-plan --observations --report; neverinsertheldoutrowsintotestedtable.
  linear-holdout-report.json:187stable,13over10%,max34.7575%;accuracygateFAILED.
  Nextdensifytraininggridwhilekeepingtheseholdoutcoordinatesindependent.
-31existingmodelhas196files/samebytes/config/sidecar/indexbut126weightfilesdiffer
  from32originalcontentmanifest. Fullbytehash371121b22244cfd5827a4ee4935f525fc4f0f114e3e4968c3b915151764f18ed.
  tensor-first-node31/32.jsonconfirmsfirstshardhassamecanonicaltensorheaderbut
  DIFFERENTpayload. Current32firstshardalso differsfromarchivedb364manifest:
  currentab335b89df4efd0b8806d460d272cd82e3eeafc2c5f2ad25f2a56080568a98b8.
  Sharedcopytaskterminated;partialdirectoryrenamed
  /tgm_rent_data/data/youxingye/model-baselines/glm52-w8a8-b3645035-rejected-20261022.
  Source/destinationoriginalsnotmodified. Originalcopyranasuserandcouldnotread
  .msc;executorerrorwasdelayeduntilotherfuturesdrained. Fixedfailurehandlingin
  copy_verified_model.py,butdonotrestartagainstoldmanifest. Current32fullhash
  runningasroot(model-current-node32-fingerprint.json)toestablishactualbaseline.
  Allpriorrealrouting/servingobservationsretainoldweightidentity;syntheticlinear
  curvesremainshape/runtime-qualified. NeedresolveweightsbeforePDserving.
- Preserveallfailedinitialsmokesinlinear-20261022. Dynamicquantreferencefixedafter
  <=1binCPUroundingdifferencewithmatching scales;separatequantchecks remain.
- Rawlinearpointtablesuseparquet;syntheticinputs/weights,source-selectedkernels.
  FullAscendGLMmodel,attention/MoE/DeepEP/resident/PD/searchareNOTcomplete.

Frozen initial shared-input md/910c-glm52-w8a8-full-adaptation.md mustremainunchanged;
currentupdatesgo tomd/910c-glm52-w8a8-production-progress.md andresearchcheckpoint.
Changing the initialscope docinvalidatescheckpointinputhashes.

## Runtime and model

Pinned image:
quay.io/ascend/sglang@sha256:5f384cab0e16a4951ed0bdc54e1280f2108aeda6b8ede90ee527d8f4b5146443
SGLang0.5.22 commit baa9befa381511cd4d56f83980951e5a45fffc6d,
CANN/HCCL9.1.0B243, Torch2.10.0+cpu, torch-npu2.10.0.post6, Python3.12.
Current31/32drivers26.1.1;historical19driver25.5.0. Actual device Ascend910_9382, arch query2201.
Do not treat as arch35 or CUDA SM.

Model source /mnt/raid/user/data/models/glm-q, mounted /models/glm-q in United.
Peer exact copy /mnt/raid/aisimulate-youxingye-20261022/glm-q.
196files,774079436873bytes; full content manifest:
b3645035a8b916c630ccab2f0d9be6b5971202e0ac47c9cf2d6dab63c1c6a9cc.
Index filename is quant_model_weights.safetensors.index.json.
Copy verified; no need to retransmit or rehash720GiB without new concerns.

United: global SGLangTP16, attentionDP2/TP8, MoEEP16/TP1, PP1 CP1 DCP1,
MTPoff, BF16activations/KV; static W8A8 attention linears; dynamic W8A8 MLPs.
78layers=3dense+75MoE,256routed experts top8 +1shared; hidden6144,
denseinter12288,expertinter2048. MLA64heads,qLoRA2048,kvLoRA512,
qkNoPE192,RoPE64,V256; indexer32heads x128,topk2048.
21full indexers (0,1,2,6,...74),57skip. Actual resident indexer modules also
appear only on full layers, but A3 allocates index KV for all78layers.

serving-setup.json records actual queue0,HCCL650,expandable segments,
multi/overlapstream flags,dispatchlimit32,and40logicalCPUs perrank.
CLI chunk16384 becomes8192 perattentionDP; maxrunning64global/32perDP.
Decode graphs8/16/24/32; prefill graph disabled; shared fusion disabled.
Context limit180000, actual usable pool52992tokens perattentionDP,
maxinput52986. Capacity is not multiplied by TP.

## IMPORTANT: HCCL v2 was withdrawn

Code review found v2 verify() ran EAGER even for graph measurements.
Historical row equality checks did not prove graph replay correctness.
All pre-v3 HCCL publications in this campaign are historical, NOT current
qualified inputs. Do not relabel/reuse old graphs or copy their timings.

dd1eb7ab fixes this:
- Actual timed eager batch/captured graph verified before and after timing.
- Separate AllReduce buffer per invocation permits finite nonzero checks,
  while timing/warmup use zeros. Warmup is batches of iterations.
- Rawv3 binds collector/framework source hashes and execution verification.
- Exporter rejectsv2; native loader requires execution_verification matching mode.
- Real2rank graph-noop injection failed as expected.64positive cases passed
  across BF16/FP16/FP32/INT32,4collectives,bothmodes,256/100663296bytes.
- Fresh full data: local320=>278qualified, peer320=>285, dual32rank80=>60.
  All720 numerical checks passed;97CV>10% points retained and excluded.
- All623 published rows reproduced by real native queries.
- 6export tests,5native table tests,390GPU parity tests passed.
- v2 rejection tested against real old table with new native binary.

Use hccl-replay-verified-20261022/published-{local,peer,dual}/ only.
Local/peer/mixed-driver placement remain distinct scopes. Dual is same measured
SuperPod, not general Ethernet/cross-SuperPod or actual PD KV handoff.
Raw JSONs are gzip archived with original-byte hashes in sealed-validation.json;
plaintext local copies remain ignored. Decompress to rerun JSON exporter.
Old v2 local campaign retained under hccl-local-serving-20261022 with
quality-status.json. Global explanation: hccl-graph-verification-correction.json.

## KV/memory integration and observations

f5349d2f:
- build_model_config now takes existing internal system identity; canonical
  compile_engine and KVCacheEstimator pass it. Task role builder already did.
- DeepSeekV32Model target-specific KV methods: all78BF16 indexer layers;
  109824bytes/token,14057472byte fixed128token sentinel, capacity rounded128.
- BaseModel fixed pool-overhead hook defaults0; BaseBackend adds it once perpool.
- Unsupported PP/CP/DCP/MTP/cache variants reject; invalid-config error cannot
  silently fall back to naive resource sizing.
- 99target/adjacent tests,5memory report tests,390GPU parity passed.
- Request admission must still use matching128-token pages.
- TOTAL resident memory model is not finished. ascend910c.yaml misc is
  intentionally empty; do not invent nccl_mem/other_mem values.

worker-memory-20261022 holds16ranks x6startup boundaries, tensor/storage inventory,
raw gzip records, report and frozen observer. Hooks only load_model,
alloc_memory_pool,init_cuda_graphs; no forward/replay hooks.
Allranks52992usabletokens,5833850880KVbytes=5.433GiB.
Rank0 model-reachable unique storage afterload52471997948bytes,4640tensors,
4565storages. Includes parameters/buffers/direct tensorattrs, not onlyweights.
lm_head and embedding each[19360,6144]BF16 (attentionTP8).
Routed weights[16,6144,4096] and[16,2048,6144]INT8;
MoE scalesBF16,offsetsFP32. Static linear inputscale/offsetBF16,deqscaleFP32,
quant_biasINT32. Do not infer resident bytes by checkpointbytes/16.

Critical memory fact:
graph phase device free dropped4587425792bytes;
Torchallocated rose23031296,reserved203423744.
Unmanaged/other allocator domains are material. Do not double-charge HCCL/graph
or infer their split from a single delta. Rank0aftergraphfree1910497280bytes.

Unprofiled controls: same frozen8192/1024requests as late state captures,
concurrency1and10,1warmup+3measured trials each.44requests allcomplete,cache0,
retractions0. Client metrics means:
C1 TTFT2534.15ms,TPOT48.09085ms,output19.79464tok/s.
C10 TTFT10759.47ms,TPOT53.39581ms,output156.63764tok/s.
Raw token arrivals and common all-admitted/output900 windows preserved.
These are client serving timings, not isolated forwards or prediction accuracy.
Private peer server stopped after completion.

## MoE state and compute evidence

real-moe-state-20261022:8workloads,28800layer/rank invocations,12662unique GMM1
states.1k/2k/4k/8k early and8kdecode concurrency2/4/8/10.
late-state-20261022: full8192/1024C1/C10, captureafter900outputs,
16ranks x3forwards x75layers each,KV9094..9097;11requests complete.
Group lists are type1 COUNTS, not cumulative offsets.
Prefill8k globalrows65536/layer; maxrank6150..15964 vsuniform4096.
Singledecode physicalbuffer4096,globalroutedrows128 duepadding.
Do not price as1request x8routes or full4096validrows.
Same input first16outputs match acrosshosts, but route distributions differ;
keep provenance separate.

moe-compute-20261022:
ascend_moe_compute.py calls actual installed W8A8GMM + DeepEPSwiGLUquant.
8observed vectors cover0..15964rows, graphbuffers4096.
176measurement groups (16smoke+64full+96reverse);2CVfailures retained.
Nonempty numerical CPU checks passed; emptycase only execution validated.
8/16weightcopy convergence<2% on GMM/chain. Sumcomponents/chain differs up5.7%.
Changing graphcalls16->32 moveszero-work chain11%.
Synthetic tensor/weight values, actualcounts; eager-source cases timed in graphs.
NOT production timing publication. Full profile/serving equivalence still needed.
Collector can be made efficient by reusing weight pools pergeometry if expanding
manycases; current protocol script regeneratesweights eachcase.

## Validation environment

CPU container aisim-910c-validation-20261021:
repoRO /workspace, current RELEASE native /tmp/aisim-native/_runtime.abi3.so.
Old pre-v3 binary backup /tmp/aisim-native/_runtime-before-hccl-v3.so.
Cargo target /tmp/aisim-target, cargo/rustfmt /root/.cargo/bin.
Build: cargo build --manifest-path /workspace/Cargo.toml --release
-p aisimulate-core --features extension-module; PYO3_PYTHON points
/usr/local/python3.12.13/bin/python3. Copy release/libaisimulate_core.so to runtime.

Use TORCH_DEVICE_BACKEND_AUTOLOAD=0 for CPUtests.
PYTHONPATH=/tmp/aisim-native:/workspace/python/aisimulate:
/workspace/python/aisimulate/src:/cli-deps
The /tmp/aisim-native/sitecustomize.py (tracked source validation_sitecustomize.py)
appends nativepath for EVERY pytest worker. Parent-only __path__.append does not
reach xdist children.
Bothparity suites with --noconftest -p no:cacheprovider -n4 passed390 in135.89s.
Use this for future relevant checks; serial earlier binaries took33min.
No CPU-performance speedup claim is made from unlike builds/testexecution.
Collector-wide suite attempts hit pre-existing Torch duplicate-triton registration;
focused Ascend tests and collector-wideRuff passed. Do not fix unrelatedtests.

## Next implementation priorities

1. Actual source-specific Ascend GLM op graph through canonical best_available.
   Generic DeepSeekV32 still constructs GPU DSA/MoE paths. Inspect __init__
   around line345; choose a scoped early branch after commongeometry setup.
   Existing native base-wrapped ops can be emitted via core.op_from_spec_json,
   like glm53flash.py; not a second predictor. Append wire variants only.
2. Define complete invocation keys and collect missing linear/attention/vector,
   MoE and DeepEP data. Static W8A8 O-proj differs by TP rank: quant_bias is
   passed only on RowParallel rank0 (linear_method_npu.py:106-121 atpinnedSHA).
   Generic GEMM(m,n,k,quant) cannot distinguish that bias/execution/layout state.
   Prefer typed Ascend ops/tables, exact runtime sources, measured brackets and
   explicit misses. Never label as CUDA/FP8 or SOL-clamp measuredNPU data using
   observed4096-cube throughput as a proven peak.
3. Attention metadata must mirror actualgraphpadding: linears mayprocessbatch8
   while only1..5requests perDP have8k/9k KV; dummy rows have runtime padlengths.
   Verify source graph input buffers/blocktable stride/pool geometry before
   construction. Earlier attention probes withall8sequences active arecontrols,
   not proof of the real1request padded workload.
4. Legacy sweeper batch is perattentionDP (global_bs=bs*attention_dp_size);
   inspect canonical replay/query normalization and attention_dp (not only
   attention_dp_size) before deriving shapes. Do not assume scheduling absent.
5. Finish resident accounting and fullforward/request accuracy. Then realPD
   KVbytes/packing/rankmaps/overlap/reservation, and predict/recommend searches.
   CollectedHCCL32 does not qualify PDtransport.
6. Keep checkpoint updated with findings/edits/results; mutableprogress only under
   research, not sharedinput artifacts. Recheck integrity on resume; no fabricated
   profile acceptance. Commit completed stages and keep fullgoalactive.
