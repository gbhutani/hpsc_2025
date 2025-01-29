x=1000000.0
s=1.0
kmax=1000
tol = 1.0e-14
for k in range(kmax):
    s0=s
    s = 0.5*(s + x/s)
    print(f"At iteration {k+1}, the value s={s}")
    delta_s = s-s0
    if(abs(delta_s)<tol):
        break
print(f"The final value is s={s}")    
