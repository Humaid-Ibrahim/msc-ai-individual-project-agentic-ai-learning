# This script allows to navigate to the right directory inside the Singularity Container

cd /app

echo "Running Socket"
echo $1

echo "Using split:"
echo $2

echo "Trial Name:"
echo $3

./run_prod_docker.sh $1 $2 $3