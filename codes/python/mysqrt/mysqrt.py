x=2.0
s=1.0
for k in range(10):
    s = 0.5*(s + x/s)
    print(f"At iteration {k+1}, the value s={s}")
print(f"The final value is s={s}")    
