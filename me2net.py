import os, sys, click, traceback

# This file deals with command line only. If the command line is parsed successfully,
# then we call one of the functions in me2net_worker.py.
#
# me2net_worker.py is imported in the functions that call it, this is done to
# avoid importing the heavy stuff unless we actually need to. This way, if you
# typed the command line wrong, you won't experience a long delay.

EPILOG = """The mask usage option (-mu, --mask-usage):

\b
  0  This is the default. Detect foreground of input image, then using the detected
     foreground mask, blend input image with neutral gray color, and save the result.
     You can specify another background color with the -bc option,
     or use an image file as the new background with the -bi option.
  1  Save input image plus foreground mask in alpha channel.
  2  Save foregound mask only. Masks are saved in grayscale PNG files.

"""


@click.group(epilog=EPILOG)
@click.version_option(version="1.1")

@click.option("-model",metavar="model",default="u2net",
    type=click.Choice(["u2net","u2netp","face"]),
    show_default=True, show_choices=True, help="select model: 'u2net', 'u2netp', or 'face'" )

@click.option("-mu","mask_usage",default='0',
    type=click.Choice(['0','1','2']),
    show_default=True, show_choices=True, help="mask usage" )

@click.option("-im","invert_mask",default=False,
    is_flag=True, show_default=True, help="invert detected foreground mask" )

@click.option("-t","threads",default=1,type=click.IntRange(1),
    show_default=True, help="number of worker threads")

@click.option("-fs","face_scale", help="scale factor for face outline",default=1,show_default=True,
              type=click.FloatRange(min=0.1,max=10))

@click.option("-bc","background_color", nargs=3, type=click.IntRange(0,255),default=[128,128,128],
    show_default=False, help="set background RGB color values, default: 128 128 128")

@click.option("-bi","background_image", help="specify a background image",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))

@click.pass_context
# not using **kwargs so I can see all options listed in one place
def cli(ctx, model, mask_usage,invert_mask,threads,background_color,background_image,face_scale):
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['model'] = model
    ctx.obj['mask_usage'] = mask_usage
    ctx.obj['invert_mask'] = invert_mask
    ctx.obj['threads'] = threads
    ctx.obj['background_color']=background_color
    ctx.obj['background_image']=background_image
    ctx.obj['face_scale']=face_scale

@cli.command(name="file", help="process one image file")
@click.argument("input_file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
@click.argument("output_file", type=click.Path(file_okay=True, dir_okay=False, writable=True))
@click.pass_context
def cmd_file(ctx,input_file,output_file) -> int:
    print(f"File input, {input_file} => {output_file} ...")
    #print(ctx.obj)
    import me2net_worker
    me2net_worker.CommonInit(ctx.obj)
    #print(ctx.obj)
    return me2net_worker.ProcessOneFile(ctx.obj,input_file,output_file)

@cli.command(name="dir", help="process image files in input directory")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.pass_context
def cmd_dir(ctx,input_dir,output_dir):
    print(f"Directory input, {input_dir} => {output_dir} ...")
    import me2net_worker
    me2net_worker.CommonInit(ctx.obj)
    return me2net_worker.ProcessOneDirectory(ctx.obj,input_dir,output_dir)


@cli.command(name="stdin", help="read RGB24 images (piped in by another program) from stdin")
@click.argument("image_width", type=click.IntRange(1))
@click.argument("image_height", type=click.IntRange(1))
@click.argument("output_specifier", type=click.STRING)
@click.pass_context
def cmd_rs(ctx,image_width,image_height,output_specifier):
    print(f"Read RGB bytes from stdin, {image_width}x{image_height} images => {output_specifier}")
    import me2net_worker
    me2net_worker.CommonInit(ctx.obj)
    return me2net_worker.ReadStdin(ctx.obj,image_width,image_height,output_specifier)
    #print(ctx.obj)


if __name__ == '__main__':
    try:
        cli(obj={},allow_interspersed_args =True,max_content_width=max(80,os.get_terminal_size().columns))
    except Exception as e:
        print(traceback.format_exc())
        sys.exit(-1)