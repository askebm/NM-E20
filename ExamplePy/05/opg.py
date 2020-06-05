from nr_python import *

eq = lambda x: x-np.cos(x)
eq_prime = lambda x: np.sin(x)+1

def make_table(k,x_est,dk):
    df = pd.DataFrame(np.vstack((x_est,dk)).T,index=k,columns=['x[k]','d[k]'])
    df.index.name='k'
    return df

## bisect
bk,bx,bdk = bisect(eq,0,np.pi/2)
bisect_df = make_table(bk,bx,bdk)

## secant
sk,sx,sdk = secant(eq,0,np.pi/2)
secant_df = make_table(sk,sx,sdk)

## false_position
fk,fx,fdk = false_position(eq,0,np.pi/2)
false_position_df = make_table(fk,fx,fdk)

## ridder
rk,rx,rdk = ridder(eq,0,np.pi/2)
ridder_df = make_table(rk,rx,rdk)

## newton
nk,nx,ndk = newton(eq,eq_prime,0)
newton_df = make_table(nk,nx,ndk)

