
function power_method(M, v)
    for i in 1:100
        v = M*v        # repeatedly creates a new vector and destroys the old v
        v /= norm(v)
    end
    
    return v, norm(M*v) / norm(v)  # or  (M*v) ./ v
end

using Gadfly
#Sparse matrix in julia works by S[I(k),J(K)]=v(k)
#in an arrow head matrix - diagnol elements and the first row and first column are non zero

n = 3
all_n_values = [i for i=1:n]
all_n_values_without_one = [i for i=2:n]
I = [all_n_values;rand(1:1,n-1);all_n_values_without_one]
J = [all_n_values;all_n_values_without_one;rand(1:1,n-1)]

#Let's create random Floating point vectors of lenght n, n-1 for the diagnol and 
#row matrices respectively. We only do this once so we get the same sort of symmetric 
#arrow head matrix all through out
float_random_x=rand(Float64,n)
float_random_x_n1 = rand(Float64,n-1)
x_lamba = rand(Float64,n)

V = [float_random_x;float_random_x_n1;float_random_x_n1]
arrow_head_matrix = sparse(I,J,V)

# Use this to make sure the arrow head matrix is correct - Sparse doesnt display the whole array
# import Base.full
# M = full(arrow_head_matrix)
println("Calling power method on a matrix of size $(n) randomly initiazed with random integers")
@time power_method(arrow_head_matrix,x_lamba)

type SymArrowFloat
    diag::Vector{Float64}
    first_row::Vector{Float64}  # without first entry
end

example_arrow_float_1 = SymArrowFloat(float_random_x,float_random_x_n1)
example_arrow_float_2 = SymArrowFloat(rand(Float64,n),rand(Float64,n-1))

import Base.show
importall Base.Operators

function Base.show(io::IO, arrow_float::SymArrowFloat)
    println(io, "Diagnol: $(arrow_float.diag) Row Entries $(arrow_float.first_row)")
    println(io, "Full Matrix: $(full(arrow_float))")
end

function full(arrow_matrix_float::SymArrowFloat)
    diagnoal = arrow_matrix_float.diag
    row_entr = arrow_matrix_float.first_row
    m = length(diagnoal)
    M = zeros(m,m)
    first_row = [diagnoal[1];row_entr]
    for i=1:m
        for j=1:m
            if i==j
                M[i,j]=diagnoal[i]
            end
        end
    end
    M[:,1]=first_row
    M[1,:]=first_row
    return M
end
#If you try to add these vectors together before definiing + it will throw the following error
# MethodError: no method matching +(::SymArrowFloat, ::SymArrowFloat)
#add_arrows = example_arrow_float_1 + example_arrow_float_2

function Base.:+(arrow_1::SymArrowFloat,arrow_2::SymArrowFloat)
    new_diag=arrow_1.diag+arrow_2.diag
    new_first_row=arrow_1.first_row+arrow_2.first_row
    return SymArrowFloat(new_diag,new_first_row)
end

function Base.:*(arrow_1::SymArrowFloat,arrow_2::SymArrowFloat)
    arrow_1_full = full(arrow_1)
    arrow_2_full = full(arrow_2)
    return arrow_1_full*arrow_2_full
end


add_arrows = example_arrow_float_1+example_arrow_float_2
mul_arrows = example_arrow_float_1*example_arrow_float_2
show(example_arrow_float_1)

println("Calculating power method for example arrow SymArrowFloat")
@time power_method(full(example_arrow_float_1),x_lamba)
#Calculate the maximum eigenvalue by using the power method


type SymArrow{T}
    diag::Vector{T}
    first_row::Vector{T}  # without first entry
end

function full{T}(arrow_matrix::SymArrow{T})
    diagnoal = arrow_matrix.diag
    row_entr = arrow_matrix.first_row
    m = length(diagnoal)
    M = zeros(m,m)
    first_row = [diagnoal[1];row_entr]
    for i=1:m
        for j=1:m
            if i==j
                M[i,j]=diagnoal[i]
            end
        end
    end
    M[:,1]=first_row
    M[1,:]=first_row
    return M
end

function +{T}(A::SymArrow{T}, B::SymArrow{T})
    new_diag=arrow_1.diag+arrow_2.diag
    new_first_row=arrow_1.first_row+arrow_2.first_row
    return SymArrow(new_diag,new_first_row)
end

function *{T}(A::SymArrow{T}, B::SymArrow{T})
    arrow_1_full = full(A)
    arrow_2_full = full(B)
    return arrow_1_full*arrow_2_full
end


#Create a BigFloat SymArrow object
x_float_diag = rand(Float32,n)
x_big_float = [BigFloat(x_float_diag[i]) for i=1:n]
x_float_row = rand(Float32,n-1)
x_big_row = [BigFloat(x_float_row[i]) for i=1:n-1]
sym_arrow_big_float = SymArrow(x_big_float,x_big_row)


#Create a Complex SymArrow object
x_float_diag = rand(Float32,n)
x_complex_float = [Complex(x_float_diag[i]) for i=1:n]
x_float_row = rand(Float32,n-1)
x_complex_row = [Complex(x_float_row[i]) for i=1:n-1]
sym_arrow_big_complex = SymArrow(x_complex_float,x_complex_row)


#Create a Rational SymArrow object
x_float_diag = rand(Float32,n)
x_rational_diag = [Rational(x_float_diag[i]) for i=1:n]
x_float_row = rand(Float32,n-1)
x_rational_row = [Rational(x_float_row[i]) for i=1:n-1]
sym_arrow_big_complex = SymArrow(x_rational_diag,x_rational_row)

println("Calculating power method for example arrow SymArrow - Float type")
Mat = SymArrow(float_random_x,float_random_x_n1)
@time power_method(full(Mat),x_lamba)

#Use workspace() to clear the previously defined full and + functions
workspace()
n = 1000
type SymArrow2{T} <: AbstractMatrix{T}
    diag::Vector{T}
    first_row::Vector{T}  # without first entry
end


import Base: size, getindex

#Without this you will receive a method error : MethodError: no method matching size(::SymArrow2{Rational{Int64}})
# Closest candidates are:
#   size(::AbstractArray{T,N}, ::Any) where {T, N} at abstractarray.jl:29
#   size(::Any, ::Integer, ::Integer, ::Integer...) where N at abstractarray.jl:30
#   size(::Char) at char.jl:13
size(A::SymArrow2{T}) where T = (length(A.diag), length(A.diag)) 

function getindex(A::SymArrow2{T}, i, j) where T
    if i == j
        return A.diag[i] 
    elseif i == 1  
        return A.first_row[j-1]
    elseif j == 1 
        return A.first_row[i-1] #Matrix is symmetric
    end 
    return zero(T)  # otherwise return zero of type T
end



import Base.full
Mat = SymArrow2(rand(Float64,n),rand(Float64,n-1))

function power_method(M, v)
    for i in 1:100
        v = M*v        # repeatedly creates a new vector and destroys the old v
        v /= norm(v)
    end
    return v, norm(M*v) / norm(v)  # or  (M*v) ./ v
end

println("Calculating power method for example arrow SymArrow2 - Float type")
@time power_method(full(Mat),rand(Float64,1000))

#They may vary when you run the file since I am assuming random numbers at the last before fixing the matrix.
# Calling power method on a matrix of size 1000 randomly initiazed with random integers 


# 0.190553 seconds (103.67 k allocations: 6.956 MiB)
# Calculating power method for example arrow SymArrowFloat
# 0.294988 seconds (113.39 k allocations: 15.410 MiB)
# Calculating power method for example arrow SymArrow - Float type
# 0.063686 seconds (5.28 k allocations: 9.429 MiB, 9.49% gc time)
# Calculating power method for example arrow SymArrow2 - Float type
# 2.153372 seconds (57.11 k allocations: 4.278 MiB)



# dhcp-18-111-23-155:Downloads manvithaponnapati$ julia Problem1.jl
# Calling power method on a matrix of size 1000 randomly initiazed with random integers
# 0.194344 seconds (103.67 k allocations: 6.956 MiB)
# Calculating power method for example arrow SymArrowFloat
# 0.292913 seconds (113.39 k allocations: 15.410 MiB)
# Calculating power method for example arrow SymArrow - Float type
# 0.066086 seconds (5.28 k allocations: 9.429 MiB, 8.51% gc time)
# Calculating power method for example arrow SymArrow2 - Float type
# 2.167233 seconds (57.11 k allocations: 4.278 MiB)


#From the implementation results it is very clear that ,
#SymArrow2{T} - has the WORST time
#SymArrow{T} - has the BEST time consistently for calculating the power method

# Why ?


# The reasons why SymArrow2 is performing less efficiently is obvious - Because we are falling 
# back on the generic functions (like full) provided by Julia - we are not utilizing the structure
# of the matrix as much as we could and there by losing the speed we could have gotten with our full
# custom full function by making use of the structure of the arrow head matrix

#Between SymArrow{T} and SymArrowFloat, SymArrow{T} is doing better in terms of memory and time. 
# Among these two SymArrow{T} is performing better than the SymArrowFloat because in SymArrow{T} it's possible
# to determine the type of the variable from the wrapper itself. Ref: http://web.mit.edu/julia_v0.5.0/.julia-3c9d75391c.amd64_ubuntu1404/share/doc/julia/html/manual/performance-tips.html
