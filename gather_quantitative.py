import numpy as np

fin = open("result_100000.txt",'r')
lines = fin.readlines()
fin.close()
numbers = [float(i.strip()) for i in lines[0].split()]
normal_numbers_gt2pred = [float(i.split()[0]) for i in lines[1:]]
normal_numbers_pred2gt = [float(i.split()[1]) for i in lines[1:]]
assert len(numbers)==21
assert len(normal_numbers_gt2pred)==90
assert len(normal_numbers_pred2gt)==90

fin = open("result_counts.txt",'r')
lines = fin.readlines()
fin.close()
V = int(round(float(lines[-1].split()[0])))
T = int(round(float(lines[-1].split()[1])))

fin = open("result_tri_angle.txt",'r')
lines = fin.readlines()
fin.close()
anumbers = [float(i) for i in lines]
assert len(anumbers)==180

print( "& %.3f & %.3f & %.3f & %.3f & %.3f & %d & %d &  & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f \\\\" %
(numbers[5]*100000, numbers[11], numbers[8], numbers[17]*100, numbers[20], V, T,
normal_numbers_gt2pred[80]*100, normal_numbers_gt2pred[30]*100, normal_numbers_gt2pred[5]*100,
normal_numbers_pred2gt[80]*100, normal_numbers_pred2gt[30]*100, normal_numbers_pred2gt[5]*100,
anumbers[10-1]*100, anumbers[20-1]*100, anumbers[30-1]*100,
) )
