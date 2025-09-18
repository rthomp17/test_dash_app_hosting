
import os
import os.path as osp
import numpy as np
import pickle
import copy as cp
import json
from PIL import Image
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import Dash, dcc, html, Input, Output, State
import base64
from io import BytesIO

def image_to_base64(img):
    """Convert PIL Image to base64 string for use in Dash."""
    if isinstance(img, str):
        # If it's a file path, load the image
        img = Image.open(img)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_base64}"

def generate_mesh_images(mesh_files, object_type):
    """Generate and save images of meshes for different object types."""
    path_configs = {
        'syn_racks': {
            'path': "../relational_ndf/src/rndf_robot/descriptions/objects/syn_racks_easy_obj/{}.obj",
            'output_dir': "mesh_images/syn_racks/"
        },
        'bowls': {
            'path': "../relational_ndf/src/rndf_robot/descriptions/objects/bowl_centered_obj_normalized/{}/models/model_normalized.obj",
            'output_dir': "mesh_images/bowls/"
        },
        'mugs': {
            'path': "../relational_ndf/src/rndf_robot/descriptions/objects/mug_centered_obj_normalized/{}/models/model_normalized.obj",
            'output_dir': "mesh_images/mugs/"
        }
    }
    
    if object_type not in path_configs:
        raise ValueError(f"Unknown object type: {object_type}")
    
    config = path_configs[object_type]
    camera_config = {"up": dict(x=0, y=1, z=0), "eye": dict(x=2.5, y=1.75, z=1)}
    
    for pcl_id in mesh_files:
        obj_file_path = config['path'].format(pcl_id)
        mesh = utils.trimesh_load_object(obj_file_path)
        
        fig = viz_utils.show_meshes_plotly(
            {"mesh": mesh.vertices},
            {"mesh": mesh.faces},
            center=True,
            axis_visible=False,
            background_visible=False,
            camera=camera_config,
            show_legend=False,
            show=False,
        )
        fig.write_image(f"{config['output_dir']}{pcl_id}.png")

def circlify_image(img):
    datas = img.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    img.putdata(newData)
    return img  

def color_adjust_image(img, cost, cost_thresh, idx, training_ids, make_translucent=False):
    img = np.array(img)
    if idx in training_ids:
        img[:, :, 0] = img[:, :, 0]//2
        img[:, :, 1] = img[:, :, 1]//2
        img[:, :, 2] = np.ones_like(img[:, :, 3]) * 200
        img[:, :, 3] = np.where(img[:,:,3] != 0, np.ones_like(img[:, :, 3]) * 50,  img[:, :, 3])
        return Image.fromarray(img)

    if cost is not None and cost >= cost_thresh:
        img[:, :, 0] = img[:, :, 0]//2
        img[:, :, 2] = img[:, :, 1]//2
        img[:, :, 1] = np.ones_like(img[:, :, 3]) * 200
    else:
        img[:, :, 1] = img[:, :, 0]//2
        img[:, :, 2] = img[:, :, 1]//2
        img[:, :, 0] = np.ones_like(img[:, :, 3]) * 200

    if make_translucent:
        if cost is not None and cost >= cost_thresh:
            img[:, :, 3] = np.where(img[:,:,3] != 0, np.ones_like(img[:, :, 3]) * 90,  img[:, :, 3])
        else:
            img[:, :, 3] = np.where(img[:,:,3] != 0, np.ones_like(img[:, :, 3]) * 50,  img[:, :, 3])
        
    return Image.fromarray(img)


def graph_objects_by_embeddings(embedding_info, pcl_ids, costs=None, 
                               cost_thresh=None, training_ids=None, graph_save_file=None):
    """Simple embedding visualization without success coloring."""
    fig = _create_base_scatter_plot(embedding_info)
    x_range, y_range = _get_embedding_ranges(embedding_info)

    for pcl_id, embedding in zip(pcl_ids, embedding_info):
        fig.add_layout_image(
            dict(
                source=circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                xref="x", yref="y", xanchor="center", yanchor="middle",
                x=embedding[0], y=embedding[1],
                sizex=x_range / 4, sizey=y_range / 4,
                sizing="contain", opacity=0.8, layer="above"
            )
        )
    return fig


def _create_base_scatter_plot(object_embeddings):
    """Create base scatter plot with transparent markers."""
    fig = px.scatter(x=object_embeddings[:, 0], y=object_embeddings[:, 1])
    fig.update_traces(marker_color="rgba(0,0,0,0)")
    return fig

def _get_embedding_ranges(object_embeddings):
    """Calculate x and y ranges for embedding space."""
    x_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])
    y_range = np.max(object_embeddings[:, 0]) - np.min(object_embeddings[:, 0])
    return x_range, y_range

def _add_image_to_plot(fig, pcl_id, embedding, cost, cost_thresh, training_ids, 
                      sizex, sizey, make_translucent=False):
    """Add a single mesh image to the plot."""
    fig.add_layout_image(
        dict(
            source=color_adjust_image(
                circlify_image(Image.open(f"mesh_images/mugs/{pcl_id}.png")),
                cost, cost_thresh, pcl_id, training_ids, make_translucent=make_translucent
            ),
            xref="x", yref="y", xanchor="center", yanchor="middle",
            x=embedding[0], y=embedding[1],
            sizex=sizex, sizey=sizey, sizing="contain", layer="above"
        )
    )

def image_grid(imgs, rows, cols):
    """Create a grid of images."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
        
def get_experiment_results():
    # Check for cached results
    cache_file = 'experiment_results_cache.json'
    if os.path.exists(cache_file):
        print(f"Loading cached experiment results from {cache_file}")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        return cached_data['results_nums'], cached_data['whole_nums'], cached_data['rndf_nums'], cached_data['lndf_nums']

    print("Computing experiment results from scratch...")
    results_dict = {}
    whole_results_dict = {}
    rndf_results_dict = {}
    lndf_results_dict = {}

    exp_type = 'mug_on_rack'#'bowl_on_mug'#
    root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/intersect_test_exp--{exp_type}_sweep_demo-exp--release_demos/'

    #root_dir = f'/home/rthomp12/relational_ndf/src/rndf_robot/eval_data/eval_data/exp--{exp_type}_upright_pose_new_demo-exp--release_demos/'
    experiment_folders = os.listdir(root_dir)
    rndf_folders = [folder for folder in experiment_folders if 'rndf' in folder]
    lndf_folders = [folder for folder in experiment_folders if 'lndf' in folder]
    part_whole_folders = [folder for folder in experiment_folders if 'rndf' not in folder and 'lndf' not in folder and folder != 'old']
    
    #part and whole results
    for exp_folder in part_whole_folders:
        # print(exp_folder)
        # objects_raw = os.listdir(root_dir+exp_folder) 

        exp_folder += '/'# + objects_raw[0] + '/'

        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]

        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/parts_based_success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in results_dict.keys():
                    results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue
                
            try:
                experiment_file = root_dir + exp_folder + folder + '/whole_success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in whole_results_dict.keys():
                    whole_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    whole_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    results_nums = {id_number:sum(results_dict[id_number])/len(results_dict[id_number]) for id_number in results_dict.keys()}
    whole_nums = {id_number:sum(whole_results_dict[id_number])/len(whole_results_dict[id_number]) for id_number in whole_results_dict.keys()}

    for exp_folder in rndf_folders:
        exp_folder += '/'
        exp_folder += os.listdir(root_dir+exp_folder)[0]
        exp_folder += '/'
        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]
        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in rndf_results_dict.keys():
                    rndf_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    rndf_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    for exp_folder in lndf_folders:
        exp_folder += '/'
        objects_raw = os.listdir(root_dir+exp_folder) 
        trial_folders = [fn for fn in objects_raw if (fn.split('_')[0] == 'trial')]
        for folder in trial_folders:
            #print(folder)
            try:
                experiment_file = root_dir + exp_folder + folder + '/success_rate_relation.npz'
                result = np.load(experiment_file, allow_pickle=True)
                
                child_id = str(result['child_id'])
                if child_id in lndf_results_dict.keys():
                    lndf_results_dict[child_id].append(1 if result['place_success'] else 0)
                else:
                    lndf_results_dict[child_id] = [1] if result['place_success'] else [0]
            except FileNotFoundError:
                continue

    rndf_nums = {id_number:sum(rndf_results_dict[id_number])/len(rndf_results_dict[id_number]) for id_number in rndf_results_dict.keys()}
    lndf_nums = {id_number:sum(lndf_results_dict[id_number])/len(lndf_results_dict[id_number]) for id_number in lndf_results_dict.keys()}

    # Cache the results
    cache_data = {
        'results_nums': results_nums,
        'whole_nums': whole_nums,
        'rndf_nums': rndf_nums,
        'lndf_nums': lndf_nums
    }

    print(f"Saving experiment results to cache file {cache_file}")
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    return results_nums, whole_nums, rndf_nums, lndf_nums


def save_graph_data_to_json(whole_data, part_data, rndf_data, lndf_data):
    """Save all graph data to individual JSON files for faster loading."""
    graphs = {
        'whole': whole_data,
        'part': part_data,
        'rndf': rndf_data,
        'lndf': lndf_data
    }

    for graph_name, data in graphs.items():
        embeddings, costs, cost_thresh, pcl_ids, training_ids = data

        # Convert numpy arrays to lists for JSON serialization
        graph_data = {
            'embeddings': embeddings.tolist(),
            'costs': costs if costs is not None else None,
            'cost_thresh': cost_thresh,
            'pcl_ids': pcl_ids,
            'training_ids': training_ids
        }

        filename = f'{graph_name}_graph_data.json'
        print(f"Saving {graph_name} graph data to {filename}")
        with open(filename, 'w') as f:
            json.dump(graph_data, f, indent=2)


def load_graph_data_from_json():
    """Load all graph data from JSON files if they exist."""
    graphs = {}
    graph_names = ['whole', 'part', 'rndf', 'lndf']

    for graph_name in graph_names:
        filename = f'{graph_name}_graph_data.json'
        if os.path.exists(filename):
            print(f"Loading {graph_name} graph data from {filename}")
            with open(filename, 'r') as f:
                graph_data = json.load(f)

            # Convert lists back to numpy arrays
            embeddings = np.array(graph_data['embeddings'])
            costs = graph_data['costs']
            cost_thresh = graph_data['cost_thresh']
            pcl_ids = graph_data['pcl_ids']
            training_ids = graph_data['training_ids']

            graphs[graph_name] = (embeddings, costs, cost_thresh, pcl_ids, training_ids)
        else:
            return None  # If any file is missing, return None to recompute all

    return graphs


def preload_images(pcl_ids_list):
    """Pre-load all images as base64 and image arrays for client-side access."""
    image_dict = {}
    all_pcl_ids = set()

    # Collect all unique PCL IDs from all lists
    for pcl_ids in pcl_ids_list:
        all_pcl_ids.update(pcl_ids)

    for pcl_id in all_pcl_ids:
        img_path = f"mesh_images/mugs/{pcl_id}.png"
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                img = img.crop((100, 100, 500, 500))
                img = img.resize((250, 250), Image.Resampling.LANCZOS)

                # Convert to numpy array for Plotly
                img_array = np.array(img)

                # Store both base64 and array data
                image_dict[pcl_id] = {
                    'base64': image_to_base64(img),
                    'array': img_array.tolist(),  # Convert to list for JSON serialization
                    'shape': img_array.shape
                }
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image_dict[pcl_id] = None
        else:
            image_dict[pcl_id] = None

    return image_dict


def create_four_graph_interactive_app(whole_data, part_data, rndf_data, lndf_data):
    """Create an interactive Dash app with all four visualizations in a 2x2 grid with synchronized hover."""

    # Create individual figures for each visualization
    def create_subplot_figure(data, title, graph_id):
        embeddings, costs, cost_thresh, pcl_ids, training_ids = data

        # Create base scatter plot with transparent markers
        fig = go.Figure()
        x_range, y_range = _get_embedding_ranges(embeddings)

        # Add non-sampled images (smaller, translucent)
        for i, (pcl_id, embedding, cost) in enumerate(zip(pcl_ids + training_ids, embeddings, costs or [None]*len(pcl_ids))):
            _add_image_to_plot(fig, pcl_id, embedding, cost, cost_thresh, training_ids,
                                x_range / 4, y_range / 4, make_translucent=True)

        # Add invisible scatter points for hover detection
        x_coords = embeddings[:, 0]
        y_coords = embeddings[:, 1]
        custom_data = pcl_ids

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=30,  # Smaller markers for grid layout
                color='rgba(0,0,0,0)',  # Completely transparent
                #line=dict(width=0)
            ),
            customdata=custom_data,
            hovertemplate=None,
            hoverinfo="none",
            name='Hover Targets',
            showlegend=False
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            hovermode='closest',
            width=500,
            height=350,
            margin=dict(l=15, r=15, t=30, b=10),
            uirevision=graph_id  # Preserve zoom/pan state
        )
        return fig, pcl_ids, costs, cost_thresh, training_ids

    # Create figures for all four visualizations
    whole_fig, whole_pcl_ids, whole_costs, whole_cost_thresh, whole_training_ids = create_subplot_figure(
        whole_data, "Whole Object Shape Warping", "whole")
    part_fig, part_pcl_ids, part_costs, part_cost_thresh, part_training_ids = create_subplot_figure(
        part_data, "Parts-Based Shape Warping", "part")
    rndf_fig, rndf_pcl_ids, rndf_costs, rndf_cost_thresh, rndf_training_ids = create_subplot_figure(
        rndf_data, "Relational Neural Descriptor Fields", "rndf")
    lndf_fig, lndf_pcl_ids, lndf_costs, lndf_cost_thresh, lndf_training_ids = create_subplot_figure(
        lndf_data, "Local Neural Descriptor Fields", "lndf")

    # Pre-load all images for client-side access
    all_pcl_ids_lists = [
        rndf_pcl_ids + rndf_training_ids,
        lndf_pcl_ids + lndf_training_ids,
        whole_pcl_ids,
        part_pcl_ids
    ]
    image_data = preload_images(all_pcl_ids_lists)

    # Create the Dash app
    app = Dash(__name__)

     # Store data for callbacks - convert to JSON for client-side access
    all_data = {
        'rndf': {'pcl_ids': rndf_pcl_ids + rndf_training_ids, 'embeddings': rndf_data[0].tolist(), 'costs': rndf_costs, 'cost_thresh': rndf_cost_thresh, 'training_ids': rndf_training_ids},
        'lndf': {'pcl_ids': lndf_pcl_ids + lndf_training_ids, 'embeddings': lndf_data[0].tolist(), 'costs': lndf_costs, 'cost_thresh': lndf_cost_thresh, 'training_ids': lndf_training_ids},
        'whole': {'pcl_ids': whole_pcl_ids, 'embeddings': whole_data[0].tolist(), 'costs': whole_costs, 'cost_thresh': whole_cost_thresh, 'training_ids': whole_training_ids},
        'part': {'pcl_ids': part_pcl_ids, 'embeddings': part_data[0].tolist(), 'costs': part_costs, 'cost_thresh': part_cost_thresh, 'training_ids': part_training_ids}
    }

    # App layout with 2x2 grid
    app.layout = html.Div([
        html.H1("Interactive t-SNE Visualizations Comparison",
                style={'textAlign': 'center', 'marginBottom': '30px'}),

        # Hidden divs to store state and data
        html.Div(id='hover-state', style={'display': 'none'}),
        html.Div(id='all-data-store', children=json.dumps(all_data), style={'display': 'none'}),
        html.Div(id='image-data-store', children=json.dumps(image_data), style={'display': 'none'}),

        html.Div([
            # Main content container with plots and image display
            html.Div([
                # Left side: plots in 2x2 grid
                html.Div([
                    # Top row
                    html.Div([
                        dcc.Graph(id='rndf-plot', figure=rndf_fig,
                                 style={'display': 'inline-block'}),
                        dcc.Graph(id='lndf-plot', figure=lndf_fig,
                                 style={'display': 'inline-block'})
                    ]),
                    # Bottom row
                    html.Div([
                        dcc.Graph(id='whole-plot', figure=whole_fig,
                                 style={'display': 'inline-block'}),
                        dcc.Graph(id='part-plot', figure=part_fig,
                                 style={'display': 'inline-block'})
                    ])
                ], style={'display': 'inline-block', 'verticalAlign': 'top',}),

                # Right side: image display using Plotly graph
                html.Div([
                    dcc.Graph(
                        id='hover-image-display',
                        figure={
                            'data': [],
                            'layout': {
                                'xaxis': {'visible': False},
                                'yaxis': {'visible': False},
                                'showlegend': False,
                                'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                'width': 270,
                                'height': 270,
                                'plot_bgcolor': 'white',
                                'paper_bgcolor': 'white'
                            }
                        },
                        style={
                            'width': '270px',
                            'height': '270px',
                            'border': '2px solid black',
                            'borderRadius': '10px',
                            'marginLeft': '20px'
                        }
                    )
                ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ])
    ])

   

    # Client-side callback to detect hover and store hovered pcl_id and indices
    app.clientside_callback(
        """
        function(rndf_hover, lndf_hover, whole_hover, part_hover, all_data_str) {
            // Parse the data from the hidden div
            var all_data = JSON.parse(all_data_str);

            // Check which input was triggered using dash_clientside.callback_context
            var ctx = dash_clientside.callback_context;
            if (!ctx.triggered.length) {
                return null;
            }

            var trigger_id = ctx.triggered[0].prop_id.split('.')[0];

            // Get hover data from the triggered graph
            var hover_data_map = {
                'rndf-plot': rndf_hover,
                'lndf-plot': lndf_hover,
                'whole-plot': whole_hover,
                'part-plot': part_hover
            };

            var hover_data = hover_data_map[trigger_id];

            if (hover_data && hover_data.points && hover_data.points[0] &&
                hover_data.points[0].customdata !== undefined) {
                var pcl_id = hover_data.points[0].customdata;

                // Find the index of this pcl_id in each graph's data
                var idxs = [];
                ['rndf', 'lndf', 'whole', 'part'].forEach(function(plot) {
                    var idx = all_data[plot].pcl_ids.indexOf(pcl_id);
                    idxs.push(idx);
                });

                return JSON.stringify({'pcl_id': pcl_id, 'indices': idxs});
            }

            return null;
        }
        """,
        Output('hover-state', 'children'),
        [Input('rndf-plot', 'hoverData'),
         Input('lndf-plot', 'hoverData'),
         Input('whole-plot', 'hoverData'),
         Input('part-plot', 'hoverData'),
         Input('all-data-store', 'children')]
    )

    # Client-side callback to update image display and add circles to all graphs
    app.clientside_callback(
        """
        function(hover_state, all_data_str, image_data_str, rndf_fig, lndf_fig, whole_fig, part_fig) {
            // Parse data from hidden divs
            var all_data = JSON.parse(all_data_str);
            var image_data = JSON.parse(image_data_str);

            // Create empty image figure for hidden state
            var empty_image_fig = {
                'data': [],
                'layout': {
                    'xaxis': {'visible': false},
                    'yaxis': {'visible': false},
                    'showlegend': false,
                    'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                    'width': 270,
                    'height': 270,
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white'
                }
            };

            if (!hover_state) {
                return [empty_image_fig, rndf_fig, lndf_fig, whole_fig, part_fig];
            }

            try {
                var hover_data = JSON.parse(hover_state);
                var pcl_id = hover_data.pcl_id;
                var indices = hover_data.indices;

                // Create image figure
                var image_fig;
                if (image_data[pcl_id] && image_data[pcl_id].base64) {
                    // Create a figure with the image
                    image_fig = {
                        'data': [{
                            'type': 'image',
                            'source': image_data[pcl_id].base64,
                            'xref': 'x',
                            'yref': 'y',
                            'x': 0,
                            'y': 0,
                            'sizex': 250,
                            'sizey': 250,
                            'sizing': 'stretch',
                            'layer': 'below'
                        }],
                        'layout': {
                            'xaxis': {
                                'visible': false,
                                'range': [0, 250]
                            },
                            'yaxis': {
                                'visible': false,
                                'range': [0, 250],
                                'scaleanchor': 'x',
                                'scaleratio': 1,
                                'autorange': 'reversed'
                            },
                            'showlegend': false,
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                            'width': 270,
                            'height': 270,
                            'plot_bgcolor': 'white',
                            'paper_bgcolor': 'white'
                        }
                    };
                } else {
                    // Create a figure with error text
                    image_fig = {
                        'data': [],
                        'layout': {
                            'xaxis': {'visible': false},
                            'yaxis': {'visible': false},
                            'showlegend': false,
                            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                            'width': 270,
                            'height': 270,
                            'plot_bgcolor': 'white',
                            'paper_bgcolor': 'white',
                            'annotations': [{
                                'text': 'Image not found for ID: ' + pcl_id,
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'showarrow': false,
                                'font': {'size': 12}
                            }]
                        }
                    };
                }

                // Update figures with circles
                var figures = [rndf_fig, lndf_fig, whole_fig, part_fig];
                var graph_names = ['rndf', 'lndf', 'whole', 'part'];
                var updated_figures = [];

                for (var i = 0; i < figures.length; i++) {
                    var fig = JSON.parse(JSON.stringify(figures[i])); // Deep copy
                    var point_index = indices[i];
                    var data_info = all_data[graph_names[i]];

                    // Remove existing circle traces
                    fig.data = fig.data.filter(function(trace) {
                        return trace.name !== 'Hover Circle';
                    });

                    // Add circle if valid index
                    if (point_index !== -1 && point_index < data_info.embeddings.length) {
                        var embedding = data_info.embeddings[point_index];
                        fig.data.push({
                            x: [embedding[0]],
                            y: [embedding[1]],
                            mode: 'markers',
                            marker: {
                                size: 40,
                                color: 'rgba(255, 0, 0, 0)',
                                line: {width: 3, color: 'red'}
                            },
                            name: 'Hover Circle',
                            showlegend: false,
                            hoverinfo: 'skip',
                            type: 'scatter'
                        });
                    }

                    updated_figures.push(fig);
                }

                return [image_fig].concat(updated_figures);

            } catch (e) {
                console.error('Error in hover display:', e);
                return [empty_image_fig, rndf_fig, lndf_fig, whole_fig, part_fig];
            }
        }
        """,
        [Output('hover-image-display', 'figure'),
         Output('rndf-plot', 'figure'),
         Output('lndf-plot', 'figure'),
         Output('whole-plot', 'figure'),
         Output('part-plot', 'figure')],
        [Input('hover-state', 'children')],
        [State('all-data-store', 'children'),
         State('image-data-store', 'children'),
         State('rndf-plot', 'figure'),
         State('lndf-plot', 'figure'),
         State('whole-plot', 'figure'),
         State('part-plot', 'figure')]
    )

    return app

def main_interactive():
    """Main function to generate interactive t-SNE visualizations."""
    # Try to load cached graph data first
    cached_graphs = load_graph_data_from_json()

    if cached_graphs is not None:
        print("Using cached graph data for faster loading!")
        whole_data = cached_graphs['whole']
        part_data = cached_graphs['part']
        rndf_data = cached_graphs['rndf']
        lndf_data = cached_graphs['lndf']
    else:
        print("Computing graph data from scratch...")
        # Configuration setup
        cfg = get_eval_cfg_defaults()
        config_fname = osp.join(
            path_util.get_rndf_config(), "eval_cfgs", "base_cfg"
        )
        if osp.exists(config_fname):
            cfg.merge_from_file(config_fname)
        else:
            log_info(f"Config file {config_fname} does not exist, using defaults")

        # Load mesh data
        mesh_names = {}
        for k, v in mesh_data_dirs.items():
            objects_raw = os.listdir(v)
            objects_filtered = [
                fn for fn in objects_raw
                if (fn.split("/")[-1] not in bad_ids[k] and "_dec" not in fn)
            ]

            segmentable_filtered = []
            for obj in objects_filtered:
                if check_mug_segmentation_exists(obj):
                    segmentable_filtered.append(obj)

            mesh_names[k] = segmentable_filtered

        # File paths and parameters
        warp_file_stamp = "20240320-032402"
        object_warp_file = f"/home/rthomp12/fewshot/part_based_warp_models/old/whole_mug_{warp_file_stamp}"
        part_names = ["cup", "handle"]
        obj_type = "mug"

        # Load canonical objects and parameters
        whole_object_canonical = pickle.load(open(object_warp_file, "rb"))
        training_ids = whole_object_canonical.metadata.training_ids

        # Load experimental results and parameters
        result_nums, whole_nums, rndf_nums, lndf_nums = get_experiment_results()
        part_params = pickle.load(open(f'/home/rthomp12/fewshot/part_based_warp_models/handle_params_{warp_file_stamp}', 'rb'))

        # Create embeddings
        cup_embeddings = np.array([param.latent for param in part_params['cup']])
        handle_embeddings = np.array([param.latent for param in part_params['handle']])
        part_embeddings = {'cup': cup_embeddings, 'handle': handle_embeddings}

        # Filter data for different methods
        def filter_data(mesh_names_dict, results_dict, embeddings_dict):
            filtered_names = [name for name in mesh_names_dict[obj_type] if name in results_dict.keys()]
            filtered_idxs = [i for i, name in enumerate(mesh_names_dict[obj_type]) if name in results_dict.keys()]
            filtered_costs = [results_dict[name] for name in filtered_names]
            filtered_embeddings = {part: embeddings_dict[part][filtered_idxs] for part in part_names}
            return filtered_names, filtered_idxs, filtered_costs, filtered_embeddings

        # Filter for parts-based results
        part_mesh_names, _, part_costs, part_part_embeddings = filter_data(mesh_names, result_nums, part_embeddings)

        # Create combined embeddings for visualization
        def create_combined_embedding(part_embeddings_dict):
            return np.concatenate([
                np.atleast_2d(part_embeddings_dict['cup'][:, 0]).T,
                np.atleast_2d(part_embeddings_dict['handle'][:, 0]).T
            ], axis=1)

        # Filter for whole object results
        whole_mesh_names, _, whole_costs, whole_part_embeddings = filter_data(mesh_names, whole_nums, part_embeddings)

        # Filter for RNDF results
        rndf_mesh_names, _, rndf_costs_list, rndf_part_embeddings = filter_data(mesh_names, rndf_nums, part_embeddings)
        rndf_training_ids = np.loadtxt('./scripts/mug_train_object_split.txt', dtype=str)
        rndf_training_idxs = [i for i, name in enumerate(mesh_names[obj_type]) if name in rndf_training_ids]
        rndf_training_embeddings = {part: part_embeddings[part][rndf_training_idxs] for part in part_names}

        # Filter for LNDF results
        lndf_mesh_names, _, lndf_costs_list, lndf_part_embeddings = filter_data(mesh_names, lndf_nums, part_embeddings)
        lndf_training_ids = np.loadtxt('./scripts/lndf_mug_train_object_split.txt', dtype=str)
        lndf_training_idxs = [i for i, name in enumerate(mesh_names[obj_type]) if name in lndf_training_ids]
        lndf_training_embeddings = {part: part_embeddings[part][lndf_training_idxs] for part in part_names}

        # Create combined embeddings for all visualizations
        whole_embedding = create_combined_embedding(whole_part_embeddings)
        part_embedding = create_combined_embedding(part_part_embeddings)
        rndf_embedding = create_combined_embedding(rndf_part_embeddings)
        lndf_embedding = create_combined_embedding(lndf_part_embeddings)

        # Prepare data tuples for each visualization
        whole_data = (whole_embedding, whole_costs, 0.8, whole_mesh_names, training_ids)
        part_data = (part_embedding, part_costs, 0.8, part_mesh_names, training_ids)
        rndf_data = (np.concatenate([rndf_embedding,create_combined_embedding(rndf_training_embeddings)]), rndf_costs_list + list(np.zeros(len(rndf_training_ids))), 0.8, rndf_mesh_names, rndf_training_ids.tolist())
        lndf_data = (np.concatenate([lndf_embedding,create_combined_embedding(lndf_training_embeddings)]), lndf_costs_list+ list(np.zeros(len(lndf_training_ids))), 0.8, lndf_mesh_names, lndf_training_ids.tolist())

        # Save the computed graph data to JSON files for future runs
        save_graph_data_to_json(whole_data, part_data, rndf_data, lndf_data)

    # Generate interactive visualization with all four graphs
    print("Generating interactive visualization with all four methods...")
    app = create_four_graph_interactive_app(whole_data, part_data, rndf_data, lndf_data)
    print("Open your browser to http://localhost:8050")
    app.run_server(debug=True, port=8050)


if __name__ == "__main__":
    main_interactive()
        