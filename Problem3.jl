
@everywhere using Images
@everywhere using Colors
@everywhere using Interact
@everywhere include("imageseams.jl")

addprocs(max(0, Sys.CPU_CORES-nprocs())) #Add all possible processor
@everywhere sleep(0.1)
# if !isfile("1_yosemite_valley_tunnel_view_2010.JPG")
#     run(`wget http://upload.wikimedia.org/wikipedia/commons/e/ec/1_yosemite_valley_tunnel_view_2010.JPG`)
# end

# img = load("1_yosemite_valley_tunnel_view_2010.JPG")

# if !isfile("self-driving-taxi-header-1200x628.jpg")
#     run(`wget http://thingsautos.com/wp-content/uploads/2017/07/self-driving-taxi-header-1200x628.jpg
# `)
# end

# img = load("self-driving-taxi-header-1200x628.jpg")


if !isfile("Wfm_stata_center.jpg")
    run(`wget http://upload.wikimedia.org/wikipedia/commons/2/25/Wfm_stata_center.jpg`)
end

img = load("Wfm_stata_center.jpg")

H = size(img,1)
W = size(img,2)
println("Size of the image - H: $(size(img,1)) px tall and W: $(size(img,2)) px")


@everywhere strips = Any[]
num_of_cores = nprocs()
width_of_strips = round(Int,W/num_of_cores) #round to nearest integer

for strip_start=1:width_of_strips:W-width_of_strips+1
    strip_end=strip_start+width_of_strips-1
    push!(strips,img[:,strip_start:strip_end])
end

# This is a utility function (you do not need to understand it) 
# which overrides Ijulia's image widget so that manipulate displays with the proper width
immutable ImgFrame
    img::ImageMeta
end
ImgFrame(a::Array{<:AbstractRGB}) = ImgFrame(ImageMeta(a))
Base.show(io::IO, m::MIME"text/html", frame::ImgFrame) = 
   write(io, """<img src="data:image/png;base64,$(stringmime(MIME("image/png"), frame.img))"/>""")

@time B = @parallel hcat for i=1:nprocs()
    process(strips[i])
end

# @time A = process(img)

workers_dim = size(B,2)
strip_width = size(B,1)

X = []
for i=1:strip_width
    D = B[i,1]
    for j=2:workers_dim
        D = hcat(D,B[i,j])
    end
    push!(X,D)
end

@manipulate for image_width=1:strip_width
    ImgFrame(X[image_width])
end

@time A = process(img)
