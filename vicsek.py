''' Demonstration of the self-ordered motion in a system of particles 
described by Vicsek et al, Phys. Rev. Lett 75, 1226-1229 (1995)

Use the ``bokeh serve`` command to run the app by executing:

    bokeh serve vicsek.py

at your command prompt. Then navigate to the URL

    http://localhost:5006/vicsek

in your browser.

Created by Martin Klein Schaarberg

'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider, Button, RadioButtonGroup, Div
from bokeh.plotting import figure

def random_particles(N,L):
    x = np.random.rand(N,2)*L
    theta = (np.random.rand(N,1)-0.5)*2*np.pi
    return x, theta

def identify_groups(x,old_groups):
    groups = np.zeros(x.shape[0])


def update_angle(x0,x,theta,eta,r):
    dx = x - x0
    in_neighbourhood = dx[:,0]**2 + dx[:,1]**2 <= r**2
    dtheta = eta*(np.random.rand(1)-0.5)
    average_theta = np.arctan2(np.mean(np.sin(theta[in_neighbourhood])),
                               np.mean(np.cos(theta[in_neighbourhood])))    
    return average_theta + dtheta

def initialize_data():
    N = NSlider.value
    L = LSlider.value
    r = rSlider.value
    x, theta = random_particles(N,L)
    source.data = dict( x=x[:,0], y=x[:,1],t=np.zeros(theta.shape),theta=theta,start_angle=theta-wedge_angle/2+np.pi,end_angle=theta+wedge_angle/2+np.pi,r=r*np.ones(theta.shape))

def update_data():
    # Get the current slider values
    eta = etaSlider.value
    nu = nuSlider.value
    L = LSlider.value
    N = NSlider.value
    r = rSlider.value

    x,theta,t  = np.column_stack((source.data['x'],source.data['y'])), source.data['theta'], source.data['t']

    # compute new positions
    t = t + dt
    theta = np.apply_along_axis(update_angle,1, x,x,theta,eta,r)
    v = nu*np.column_stack((np.cos(theta),np.sin(theta)))
    x = x + v*dt

    # density and average normalized velocity
    rho = N/L**2
    nu_a = 1/(N*nu)*np.linalg.norm(np.sum(v,axis=0))

    # apply periodic boundary condition
    x[x>L] -= L
    x[x<0] += L
    
    # update plot
    source.data = dict( x=x[:,0], y=x[:,1],t=t,theta=theta,start_angle=theta-wedge_angle/2+np.pi,end_angle=theta+wedge_angle/2+np.pi,r=r*np.ones(theta.shape))
    plot.title.text = 'Time: %d, Density: %.2g, Average velocity: %.2f' % (t[0],rho, nu_a)

def toggle_draw(new):
    if toggleDraw.active == 0:
        particles.glyph.fill_alpha = particle_fill_alpha
        particles.glyph.line_alpha = 1
        circles.glyph.fill_alpha = 0
        circles.glyph.line_alpha = 0   
    else:
        particles.glyph.fill_alpha = 0
        particles.glyph.line_alpha = 0
        circles.glyph.fill_alpha = circle_fill_alpha
        circles.glyph.line_alpha = 1

def update_axis(attr,old,new):
    L = LSlider.value
    plot.x_range.end = L
    plot.y_range.end = L

def add_remove_particles(attr,old,new):
    global x, theta
    N = NSlider.value
    L = NSlider.value
    if N <= len(theta):
        x = x[:N,:]
        theta = theta[:N]
    else:
        x_new, theta_new = random_particles(N - len(theta),L)
        x = np.concatenate((x,x_new),axis=0)
        theta = np.concatenate((theta,theta_new),axis=0)

# Set up defaults
nu = 0.1
eta = 0.1
N = 300
L = 25

# fixed parameters
dt = 1
r = 1
wedge_radius = L/50
wedge_angle = np.radians(30)
particle_fill_alpha = 0.5
circle_fill_alpha = 0.1

# Set up plot
source = ColumnDataSource(data=dict(x=[], y=[],t=[],theta=[],start_angle=[],end_angle=[],r=[]))
plot = figure(plot_height=400, plot_width=400, 
              tools="",logo=None,
              x_range=[0, L], y_range=[0,L])
particles = plot.wedge('x', 'y', wedge_radius,'start_angle','end_angle',source=source,radius_units='data',fill_alpha=particle_fill_alpha)
circles = plot.circle('x', 'y',radius='r',source=source,fill_alpha=0,line_alpha=0)
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None
plot.background_fill_color = "#000000"
plot.background_fill_alpha = 0.01
plot.outline_line_width = 2
plot.outline_line_alpha = 1
plot.outline_line_color = "#111111"
#plot.xaxis.minor_tick_line_color = None
#plot.yaxis.minor_tick_line_color = None
plot.axis.visible = False

# Set up widgets
titleDiv = Div(text="""<h1>Self-ordered motion in a system of particles</h1>""")
descriptionDiv = Div(text="""<p>This app demonstrates the emergence of self-ordered motion in a particle system governed by one simple rule:</p>
<p><em>At each time step a given particle driven with a constant absolute velocity assumes the average direction of motion of the particles in its neighborhood of radius r with some random perturbation added.</em></p>
<p>For more details, see Vicsek <em>et. al.</em>, Phys. Rev. Lett. <b>75</b>, 1226-1229 (1995)</p>
<p style="font-size:80%">This app was built using <a href="bokeh.pydata.org">Bokeh</a> by <a href="http://www.martinkleinschaarsberg.nl">Martin Klein Schaarsberg</a>""")
nuSlider = Slider(title="Velocity", value=nu, start=0, end=1, step=0.01)
etaSlider = Slider(title="Perturbation", value=eta, start=0, end=2*np.pi, step=0.1)
rSlider = Slider(title="Interaction radius", value=r, start=0, end=10, step=0.1)
NSlider = Slider(title="Number of particles", value=N, start=50, end=500, step=50)
LSlider = Slider(title="Box size", value=L, start=1, end=100, step=1)
resetButton = Button(label="Reset particles",button_type='primary')
toggleDraw = RadioButtonGroup(labels=["Show particles","Show interaction radius"], active=0)

# Set slider callbacks
LSlider.on_change('value', update_axis)
NSlider.on_change('value', add_remove_particles)
resetButton.on_click(initialize_data)
toggleDraw.on_click(toggle_draw)

# Set up layouts and add to document
initialize_data()
inputs = widgetbox(titleDiv,nuSlider,etaSlider,rSlider,NSlider,LSlider,resetButton,toggleDraw,descriptionDiv)
curdoc().add_root(row(plot,inputs, width=800))
curdoc().title = "Self-ordered motion in a system of particles"
curdoc().add_periodic_callback(update_data, 10)