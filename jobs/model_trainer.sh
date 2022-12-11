#!/bin/bash
source /work/LAS/weile-lab/mdrahman/SliceLevelVulnerabilityDetection/env/bin/activate
sbatch << EOT
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --output="/work/LAS/weile-lab/mdrahman/TransformerExplainability/jobs/logs/"$1$2"_Model_Trainer.out"
#SBATCH --error="/work/LAS/weile-lab/mdrahman/TransformerExplainability/jobs/logs/"$1$2"_Model_Trainer.err"
#SBATCH --job-name=$2$1
#SBATCH --mail-user=mdrahman@iastate.edu
#SBATCH --mail-type=FAIL,END

cd /work/LAS/weile-lab/mdrahman/TransformerExplainability/code
python run.py --dataset $1 --batch $2
EOT
