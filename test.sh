python run.py --runs 5 --gpu 1 --dataset MSL --input_channels 55 --output_channels 55  --temporal_d_model 55 --spatio_d_model 55
python run.py --runs 5 --gpu 1 --dataset SWAN --input_channels 38 --output_channels 38  --temporal_d_model 38 --spatio_d_model 38 --anomaly_ratio 0.9
python run.py --runs 5 --gpu 1 --dataset GECCO --input_channels 9 --output_channels 9 --spatio_d_model 9 --temporal_d_model 9
python run.py --runs 5 --gpu 1 --dataset PSM --input_channels 25 --output_channels 25 --spatio_d_model 25 --temporal_d_model 25
python run.py --runs 5 --gpu 1 --dataset SMAP --input_channels 25 --output_channels 25 --spatio_d_model 25 --temporal_d_model 25
python run.py --runs 5 --gpu 1 --dataset SMD --input_channels 38 --output_channels 38 --spatio_d_model 38 --temporal_d_model 38 --anomaly_ratio 0.5
python run.py --runs 5 --gpu 1 --dataset SWaT --input_channels 51 --output_channels 51 --spatio_d_model 51 --temporal_d_model 51 --anomaly_ratio 0.1