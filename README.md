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





library( microbenchmark )
library( SPMatrix )

M1 = matrix( sample( 100**2 ), ncol = 100 )
M2 = matrix( sample( 100**2 ), nrow = 100 )

SetSequential( )
#SetMulticore( )
#SetGPU()

a = SPMatrix( 100, 100 )
b = SPMatrix( 100, 100 )
FillSPMatrix( a )
FillSPMatrix( b )

result = microbenchmark( SPMultiplication( a, b ),
                         M1 %*% M2,
                         times = 100L )

result
