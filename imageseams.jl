@everywhere using Images
@everywhere using Colors
@everywhere using Interact

# brightness of a color is the sum of the r,g,b values (stored as float32's)
@everywhere brightness(c::AbstractRGB) = (c.r + c.g + c.b)

# Brightness of the Image - Returns a image with zero row and column padded
@everywhere function brightness(I::Array{<:AbstractRGB})
    h, w = size(I)
    b = brightness.(I)
    # zero borders
    zh = fill(0f0,   1, w)  # horizontal zero 
    zv = fill(0f0, h+2, 1)  # vertical zero 
    [zv [zh; b; zh] zv] 
 end

# the 3x3 stencil for energy
@everywhere function stencil(b)
    x_energy = b[1,1] + 2b[2,1] + b[3,1] - b[1,3] - 2b[2,3] - b[3,3]
    y_energy = b[1,1] + 2b[1,2] + b[1,3] - b[3,1] - 2b[3,2] - b[3,3]
    sqrt(x_energy^2 + y_energy^2)
end

# energy of an array of brightness values 
# input: assumed zero borders
# output: left and right set to ∞
@everywhere function energy(b)
    h, w = size(b)
   #@time e = [float32(stencil( @view b[y-1:y+1, x-1:x+1] )) for y=2:h-1,x=2:w-1]
    e = zeros(Float32,h-2,w-2)
    for y=2:h-1,x=2:w-1
        e[y-1,x-1] = stencil( @view b[y-1:y+1, x-1:x+1] )
    end
    
    infcol = fill(Inf64, h-2, 1)
    hcat(infcol, e, infcol)
end


#  e (row                  e[x,y] 
#  dirs:                ↙   ↓   ↘       <--directions naturally live between the rows
#  e (row y+1): e[x-1,y+1] e[x,y+1]  e[x+1,y+1]     
# Basic Comp:   e[x,y] += min( e[x-1,y+1],e[x,y],e[x+1,y])
#               dirs records which one from (1==SW,2==S,3==SE)

# Take an array of energies and work up from
# bottom to top accumulating least energy down
@everywhere function least_energy(e)
# individual energies go in
# cumulative energies and directions come out
    
    h, w = size(e)
    dirs = zeros(UInt8, h-1, w-2)
       # w-2 because we don't need the infs
       # h-1 because arrows are between rows
     
    for y = h-1:-1:1, x = 2:w-1 
          
        # s, dirs[y,x-1] = findmin(e[y+1,x.+[-1, 0, 1]]) # findmin gets the min and the index
       # s, dirs[y, x-1] = findmin(@view e[y+1,x-1:x+1]) 
        a = e[y+1,x-1]
        b = e[y+1,x]
        c = e[y+1,x+1]
        
        if  (a<b)
            if (a<c)
              e[y,x] += a
              dirs[y,x-1] = 1
            else
               e[y,x] += c
               dirs[y,x-1] =3
            end
        else
            if (b<c)
              e[y,x] += b  
              dirs[y,x-1] = 2
            else
               e[y,x] += c
               dirs[y,x-1] =3
            end 
        end
        #e[y,x] += s   #  add in current energy +  smallest from below   
    end
    (@view e[1,2:w-1]), dirs  # return top row without infinities and dirs
end


@everywhere function get_seam(dirs,x)
    seam = fill(0,1+size(dirs,1))
    seam[1] = x
     for y = 1:size(dirs,1)
        seam[y+1] = seam[y] + dirs[y,seam[y]] - 2
        
    end
    return seam
end

@everywhere function mark_seam(img, seam, color=RGB(1,1,1))
    img2 = copy(img)
     for y = 1:(length(seam)-1)        
        img2[y, seam[y]]=color
    end
    img2
end

@everywhere function manipulate_seam(url::String)
    fn = split(url, "/")[end]
    
    if !isfile(fn)
        download(url,fn)
    end
    
    img = load(fn)
    _, dirs  = least_energy(energy(brightness(img)))
    
    @manipulate for x = 1:size(img, 2)
        mark_seam(img, get_seam(dirs, x))
    end
end


@everywhere function minseam(img)
    b = brightness(img)
    e1 = energy(b)
    e, dirs = least_energy(e1)
    #e, dirs = least_energy(energy(brightness(img)))
    x = indmin(e)
    seam = get_seam(dirs, x)
end
    

@everywhere function carve(img, seam)
    h, w = size(img)
    newimg = img[:,1:w-1]         # one pixel less wide

    for y = 1:h
        s = seam[y]
        #newimg[y,:] = @view img[y, [1:s-1; s+1:size(img,2)] ] # delete pixel x=s on row y
        newimg[y, 1:s-1] .= @view img[y, 1:s-1]
        newimg[y, s:end] .= @view img[y, s+1:end]
    end

    newimg
 end
    
@everywhere carve(img) = carve(img, minseam(img))  

@everywhere function carve(img, n::Int)
    img2 = copy(img)
    
    for i=1:n
        img2 = carve(img2)
    end
    
    img2
end

@everywhere function process(img)

    A = [img]  
    
    for i=1:size(img,2)-1
#     for i=1:10 
        push!(A, carve(A[end]))
        if(rem(i,10)==0) || i==size(img,1)-1 print(i, " ")
        end
    end
    return A
end


