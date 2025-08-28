conda activate /cwork/pyl10/projects/tatalab/cellSAM_env 

python simple_main.py --input /cwork/pyl10/projects/tatalab/work/inputs/tissue_dapi_fullres.tif --output fullres2 --expected-cores 20 --downsample 128 --skip-masking


python simple_main.py --input work/inputs/tissue_hires_image.png --output hires --expected-cores 9 --downsample 6 --skip-masking


python divide_tsv_by_masks.py --tsv inputs/rko4_cellseg.tsv --pipeline-output fullres/tissue_dapi_fullres_processed 




IMAGE=/opt/apps/containers/community/pyl10/valis_6.sif

apptainer exec --bind /cwork/pyl10/projects/tatalab:/workspace "$IMAGE" \                                                                      python /workspace/myvalis/valis_cli.py align \
    /workspace/inputs/core_03_stain.png \
    /workspace/inputs/core_01_dapi.png \
    /workspace/inputs/rko4_cellseg.tsv \
    --workdir /workspace/valis_work \
    --output /workspace/valis_transformed.tsv


apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --bind "$PWD:/workspace" unified_pipeline_gpu.sif python unified_pipeline_refactored.py --config /workspace/inputs/config.yaml