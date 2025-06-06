#!/bin/bash

makes=(
    Honda Volvo Genesis Mercedes-Benz
    Peugeot Lincoln BMW Tesla Infiniti
    Bentley Jeep Kia SsangYong Nissan
    Land_Rover Lexus Renault Cadillac Audi 
    Chevrolet MINI Ford Jaguar Hyundai 
    Maserati Toyota Volkswagen Porsche
)

for make in "${makes[@]}"; do
    nohup python train.py --make "${make}" --classifier mldecoder > "mldecoder_${make}.log" 2>&1
done