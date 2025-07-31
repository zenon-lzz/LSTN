python run.py --runs 5 --dataset MSL --input_channels 55 --output_channels 55 --spatio_d_model 55 --batch_size 256 --top_k 6 --dropout 0.3 --temporal_d_model 256 --temporal_encoder_layers 3 --spatio_encoder_layers 3
python run.py --runs 5 --dataset SWAN --input_channels 38 --output_channels 38 --anomaly_ratio 0.9 --spatio_d_model 55 --batch_size 256 --top_k 6 --dropout 0.3 --temporal_d_model 256 --temporal_encoder_layers 3 --spatio_encoder_layers 3
@REM python run.py --runs 5 --dataset GECCO --input_channels 9 --output_channels 9 --spatio_d_model 9 --temporal_d_model 9
@REM python run.py --runs 5 --dataset PSM --input_channels 25 --output_channels 25 --spatio_d_model 25 --temporal_d_model 25
@REM python run.py --runs 5 --dataset SMAP --input_channels 25 --output_channels 25 --spatio_d_model 25 --temporal_d_model 25
python run.py --runs 5 --dataset SMD --input_channels 38 --output_channels 38 --spatio_d_model 38 --temporal_d_model 38 --anomaly_ratio 0.5
python run.py --runs 5 --dataset SWaT --input_channels 51 --output_channels 51 --spatio_d_model 51 --temporal_d_model 51 --anomaly_ratio 0.1