# SPMatrix

library( SPMatrix )

#SetSequential( )
#SetMulticore( )
SetGPU( )

a = SPMatrix( 10, 10 )
b = SPMatrix( 10, 10 )
FillSPMatrix( a )
FillSPMatrix( b )
PrintSPMatrix( a )
PrintSPMatrix( b )
SPAddition( a, b )
PrintSPMatrix( a )
PrintSPMatrix( b )





library( SPMatrix )

#SetSequential( )
#SetMulticore( )
SetGPU( )

a = SPMatrix( 10, 10 )
b = SPMatrix( 10, 10 )
FillSPMatrix( a )
FillSPMatrix( b )
PrintSPMatrix( a )
PrintSPMatrix( b )
SPMultiplication( a, b )
PrintSPMatrix( a )
PrintSPMatrix( b )






library( microbenchmark )
library( SPMatrix )

M1 = matrix( sample( 2000**2 ), ncol = 2000 )
M2 = matrix( sample( 2000**2 ), nrow = 2000 )

#SetSequential( )
#SetMulticore( )
SetGPU()

a = SPMatrix( 2000, 2000 )
b = SPMatrix( 2000, 2000 )
FillSPMatrix( a )
FillSPMatrix( b )

result = microbenchmark( SPMultiplication( a, b ),
                         M1 %*% M2,
                         times = 30L )

result
