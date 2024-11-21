import numpy as np
import pandas as pd
import circlify
import squarify
from copy import deepcopy

from bokeh.plotting import figure
from bokeh.models.sources import ColumnDataSource
from bokeh.models import LabelSet, Span

def generate_mini_arcs_start_end_angles(value, major_tick_interval):
    start_angles = [major_tick_interval * (i-0.1) for i in range(1, int(value / major_tick_interval) + 1)]
    start_angles.append(value)
    end_angles = [major_tick_interval * i for i in range(int(value / major_tick_interval) + 2)]
    return list(zip(start_angles, end_angles))

def repeat_list(A, l):
    repeated_list = []
    while len(repeated_list) < l:
        repeated_list.extend(A)
    return repeated_list[:l]


def arc_bar_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    # title: str = "Arc Bar Chart",
    absolute_values_spiral_grid_lines_visibility: bool = False,
    fractional_values_radial_grid_lines_visibility: bool = True,
    total_values_circular_grid_lines_visibility: bool = False,
    b_fractional_ticks: bool = True,
    bool_show_legend: bool = False,
):

    padding = 0.2
    major_tick_interval = pow(10,np.ceil(np.log10(df[total_values_col_name].max()))-2)
    grid_line_color = "#e5e5e5"
    minor_tick_interval = major_tick_interval / 5

    #
    # Generating the figure
    #
    max_total_value = df[total_values_col_name].max()
    max_radius = max_total_value * (1 + padding)
    max_radius_plot = max_radius * 1.25
    p = figure(
        height=500,
        width=500,
        # title=title + " (Each complete mini-arc represents "+ str(major_tick_interval) + " units)",
        # x_axis_label="population", # Bokeh is not showing axis label in middle of chart
        x_range=(-max_radius_plot, max_radius_plot),
        y_range=(-max_radius_plot, max_radius_plot),
    )
    p.yaxis.visible = False
    p.xaxis.fixed_location = 0
    p.xaxis.bounds = (0, max_radius)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None


    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)



    #
    # Drawing grid lines
    #
    if fractional_values_radial_grid_lines_visibility:
        p.segment(
            x0=0,
            y0=0,
            x1=[max_radius * 1.02 * np.cos(theta) for theta in [0, -np.pi/2, -np.pi]],
            y1=[max_radius * 1.02 * np.sin(theta) for theta in [0, -np.pi/2, -np.pi]],
            color=grid_line_color,
            level="underlay",
        )
    if total_values_circular_grid_lines_visibility:
        x_major_ticks_values = np.arange(
            major_tick_interval, max_radius, major_tick_interval
        )
        p.ellipse(
            x=0,
            y=0,
            width=x_major_ticks_values * 2,
            height=x_major_ticks_values * 2,
            line_color=grid_line_color,
            fill_color=None,
            level="underlay",
        )
    if absolute_values_spiral_grid_lines_visibility:
        resolution = 3600
        angles = np.arange(
            np.pi * 2 / resolution,
            np.pi * 2 + np.pi * 2 / resolution,
            np.pi * 2 / resolution,
        )
        tick_radii = np.arange(minor_tick_interval, max_radius, minor_tick_interval)
        p.multi_line(
            xs=[
                [(tr * 2 * np.pi * np.cos(theta)) / theta for theta in angles]
                for tr in tick_radii
            ],
            ys=[
                [(tr * 2 * np.pi * np.sin(theta)) / theta for theta in angles]
                for tr in tick_radii
            ],
            line_color=grid_line_color,
            level="underlay",
        )
        # trimming the spirals to bound it within the polar coordinate circle
        p.multi_polygons(
            xs=[
                [
                    [
                        [
                            -max_radius_plot,
                            max_radius_plot,
                            max_radius_plot,
                            -max_radius_plot,
                        ],
                        [max_radius * np.cos(theta) for theta in angles],
                    ]
                ]
            ],
            ys=[
                [
                    [
                        [
                            -max_radius_plot,
                            -max_radius_plot,
                            max_radius_plot,
                            max_radius_plot,
                        ],
                        [max_radius * np.sin(theta) for theta in angles],
                    ]
                ]
            ],
            fill_color="white",
            line_color=None,
            level="underlay",
        )



    df_mini_arcs = deepcopy(df)
    df_mini_arcs["angles_pair"] = df_mini_arcs.apply(
        lambda row: 
        generate_mini_arcs_start_end_angles(
            row[absolute_values_col_name], 
            major_tick_interval
            ), axis=1
        )
    df_mini_arcs = df_mini_arcs.explode('angles_pair').reset_index(drop=True)
    df_mini_arcs[['start_angle', 'end_angle']] = pd.DataFrame(df_mini_arcs['angles_pair'].tolist(), index=df_mini_arcs.index)
    df_mini_arcs.drop('angles_pair', axis=1, inplace=True)
    df_mini_arcs["start_angle"] = (df_mini_arcs["start_angle"]/df_mini_arcs["whole"])*(-2 * np.pi) + np.pi*0.5
    df_mini_arcs["end_angle"] = (df_mini_arcs["end_angle"]/df_mini_arcs["whole"])*(-2 * np.pi)+ np.pi*0.5
    df_mini_arcs["radius"] = df_mini_arcs[total_values_col_name]
    cds_mini_arcs = ColumnDataSource(df_mini_arcs)

    df_bg_mini_arcs = deepcopy(df)
    df_bg_mini_arcs["angles_pair"] = df_bg_mini_arcs.apply(
        lambda row: 
        generate_mini_arcs_start_end_angles(
            row[total_values_col_name], 
            major_tick_interval
            ), axis=1
        )
    df_bg_mini_arcs = df_bg_mini_arcs.explode('angles_pair').reset_index(drop=True)
    df_bg_mini_arcs[['start_angle', 'end_angle']] = pd.DataFrame(df_bg_mini_arcs['angles_pair'].tolist(), index=df_bg_mini_arcs.index)
    df_bg_mini_arcs.drop('angles_pair', axis=1, inplace=True)
    df_bg_mini_arcs["start_angle"] = (df_bg_mini_arcs["start_angle"]/df_bg_mini_arcs["whole"])*(-2 * np.pi) + np.pi*0.5
    df_bg_mini_arcs["end_angle"] = (df_bg_mini_arcs["end_angle"]/df_bg_mini_arcs["whole"])*(-2 * np.pi)+ np.pi*0.5
    df_bg_mini_arcs["radius"] = df_bg_mini_arcs[total_values_col_name]
    cds_bg_mini_arcs = ColumnDataSource(df_bg_mini_arcs)

    p.arc(
        source=cds_bg_mini_arcs,
        radius="radius",
        start_angle="start_angle",
        end_angle="end_angle",
        line_color=color_col_name,
        x=0,
        y=0,
        line_width=2,
        alpha=0.3,
    )
    if bool_show_legend:
        p.arc(
            source=cds_mini_arcs,
            radius="radius",
            start_angle="start_angle",
            end_angle="end_angle",
            line_color=color_col_name,
            legend_field=id_col_name,
            x=0,
            y=0,
            line_width=5
        )
    else:
        p.arc(
            source=cds_mini_arcs,
            radius="radius",
            start_angle="start_angle",
            end_angle="end_angle",
            line_color=color_col_name,
            x=0,
            y=0,
            line_width=5
        )

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    df_dashed_line_to_outer_ring = deepcopy(df)
    df_dashed_line_to_outer_ring["angle"] = (df_dashed_line_to_outer_ring[absolute_values_col_name]/df_dashed_line_to_outer_ring[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_dashed_line_to_outer_ring["cos_angle"] = df_dashed_line_to_outer_ring.apply(
        lambda row: round(np.cos(row["angle"]),2), axis=1 
    )
    df_dashed_line_to_outer_ring["sin_angle"] = df_dashed_line_to_outer_ring.apply(
        lambda row: round(np.sin(row["angle"]),2), axis=1 
    )
    df_dashed_line_to_outer_ring["x0"] = df_dashed_line_to_outer_ring[total_values_col_name] * df_dashed_line_to_outer_ring["cos_angle"]
    df_dashed_line_to_outer_ring["y0"] = df_dashed_line_to_outer_ring[total_values_col_name] * df_dashed_line_to_outer_ring["sin_angle"]
    df_dashed_line_to_outer_ring["x1"] = (max_radius * 1.01) * df_dashed_line_to_outer_ring["cos_angle"]
    df_dashed_line_to_outer_ring["y1"] = (max_radius * 1.01) * df_dashed_line_to_outer_ring["sin_angle"]
    cds_dashed_line_to_outer_ring = ColumnDataSource(df_dashed_line_to_outer_ring)

    p.segment(
        source=cds_dashed_line_to_outer_ring,
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_dash="dashed",
        line_cap='round',
        alpha=0.3,
        line_width=2,
    )

    p.scatter(
        source=cds_dashed_line_to_outer_ring,
        x="x0",
        y="y0",
        color=color_col_name,
        size=10,
    )

    p.scatter(
        source=cds_dashed_line_to_outer_ring,
        x="x1",
        y="y1",
        color=color_col_name,
        size=10,
    )


    # df_lollipop_arc = deepcopy(df)
    # df_lollipop_arc["start_angle"] = df_lollipop_arc.apply(lambda row: min(row[absolute_values_col_name]+0.03*major_tick_interval, row[total_values_col_name]), axis=1)
    # df_lollipop_arc["end_angle"] = df_lollipop_arc.apply(lambda row: max(row[absolute_values_col_name]-0.03*major_tick_interval, 0), axis=1)
    # df_lollipop_arc["start_angle"] = (df_lollipop_arc["start_angle"]/df_lollipop_arc[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    # df_lollipop_arc["end_angle"] = (df_lollipop_arc["end_angle"]/df_lollipop_arc[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    # df_lollipop_arc["radius"] = df_lollipop_arc[total_values_col_name]
    # cds_df_lollipop_arc = ColumnDataSource(df_lollipop_arc)
    # p.arc(
    #     source=cds_df_lollipop_arc,
    #     radius="radius",
    #     start_angle="start_angle",
    #     end_angle="end_angle",
    #     line_color=color_col_name,
    #     legend_field=id_col_name,
    #     x=0,
    #     y=0,
    #     line_width=20,
    # )


    return p


def scatterplot(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(0, df["part"].max()*1.1), y_range=(0,1.04))
    p.toolbar.logo = None
    p.toolbar_location = None
    df_scatter = deepcopy(df)
    df_scatter["fractional_values"] = df_scatter[absolute_values_col_name]/df_scatter[total_values_col_name]
    cds_scatter = ColumnDataSource(df_scatter)
    p.scatter(x=absolute_values_col_name, y="fractional_values", size=15, color=color_col_name, source=cds_scatter)
    hline = Span(location=1, dimension='width', line_color='black', line_dash="dashed", line_width=2)
    p.renderers.extend([hline])
    return p


def progress_bar_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=df[id_col_name], y_range=(0,df[total_values_col_name].max()*1.02))
    p.xgrid.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    progress_bar_chart_df = deepcopy(df)
    progress_bar_chart_cds = ColumnDataSource(progress_bar_chart_df)

    p.vbar(source = progress_bar_chart_cds, x=id_col_name, top=total_values_col_name, color=color_col_name, width=0.85, alpha=0.3)
    p.vbar(source = progress_bar_chart_cds, x=id_col_name, top=absolute_values_col_name, color=color_col_name, width=0.85)
    return p


def bar_plus_angle_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-0.6, len(df.index)-0.4), y_range=(0,df[absolute_values_col_name].max()*1.2))
    p.xgrid.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: round(np.cos(row["angle"]),2), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: round(np.sin(row["angle"]),2), axis=1 
    )
    df_p["x0"] = df_p.index
    df_p["xs"] = df_p.apply(lambda row: [row["x0"], row["x0"], row["x0"] + 0.4 * row["cos_angle"]],axis=1)
    df_p["ys"] = df_p.apply(lambda row: [0, row[absolute_values_col_name], row[absolute_values_col_name] + (row[absolute_values_col_name] * 0.1) * row["sin_angle"]],axis=1)

    cds_p = ColumnDataSource(df_p)

    p.multi_line(
        source=cds_p,
        xs="xs",
        ys="ys",
        line_color=color_col_name,
        line_width=5,
    )

    p.scatter(x="x0", y=absolute_values_col_name, color=color_col_name, marker="+", size=50, source=cds_p)

    return p


def T_bar_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-0.6, len(df.index)-0.4), y_range=(0,df[absolute_values_col_name].max()*1.2))
    p.xgrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p["x0"] = df_p.index-0.5
    df_p["x"] = df_p.index
    df_p["x1"] = df_p.index+0.5
    df_p["x1_fractional"] = df_p["x0"] + df_p[absolute_values_col_name]/df_p[total_values_col_name]
    # df_p["xs"] = df_p.apply(lambda row: [row["x"], np.nan, row["x0"], row["x0"] + row[absolute_values_col_name]/row[total_values_col_name]],axis=1)
    # df_p["ys"] = df_p.apply(lambda row: [0, row[absolute_values_col_name], row[absolute_values_col_name], row[absolute_values_col_name]],axis=1)

    cds_p = ColumnDataSource(df_p)

    # p.multi_line(
    #     source=cds_p,
    #     xs="xs",
    #     ys="ys",
    #     line_color=color_col_name,
    #     line_width=5,
    # )

    p.segment(
        source=cds_p,
        x0="x",
        y0=0,
        x1="x",
        y1=absolute_values_col_name,
        line_color=color_col_name,
        line_width=5,
    )

    p.segment(
        source=cds_p,
        x0="x0",
        y0=absolute_values_col_name,
        x1="x1_fractional",
        y1=absolute_values_col_name,
        line_color=color_col_name,
        line_width=10,
    )

    p.segment(
        source=cds_p,
        x0="x0",
        y0=absolute_values_col_name,
        x1="x1",
        y1=absolute_values_col_name,
        alpha=0.2,
        line_color=color_col_name,
        line_width=10,
    )

    return p


def glyph_pie_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p = df_p.sort_values(by=total_values_col_name)

    # compute circle positions
    circles = circlify.circlify(
        df_p[total_values_col_name].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    # # reverse the order of the circles to match the order of data
    # circles = circles[::-1]


    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    cds_p = ColumnDataSource(df_p)
    p.circle(x="center_x", y="center_y", radius="radius", color=color_col_name, alpha=0.3, source=cds_p)
    p.wedge(x="center_x", y="center_y", radius="radius", start_angle = np.pi/2, end_angle = "angle", color=color_col_name, direction="clock", source=cds_p)
    return p


def glyph_ellipse_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p[total_values_col_name+"_squared"] = df_p[total_values_col_name]*df_p[total_values_col_name]
    df_p = df_p.sort_values(by=total_values_col_name+"_squared")

    # compute circle positions
    circles = circlify.circlify(
        df_p[total_values_col_name+"_squared"].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    df_p["width"] = df_p["radius"]*2
    df_p["height_whole"] = df_p["radius"].min()*2
    df_p["height_part"] = df_p["height_whole"]*df_p[absolute_values_col_name]/df_p[total_values_col_name]
    cds_p = ColumnDataSource(df_p)
    p.ellipse(x="center_x", y="center_y", width="width", height="height_whole",color=color_col_name, alpha=0.3, source=cds_p)
    p.ellipse(
            x="center_x",
            y="center_y",
            width="width",
            height="height_part",
            color=color_col_name,
            source=cds_p
        )
    return p


def glyph_bubble_gauge_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p = df_p.sort_values(by=absolute_values_col_name)

    # compute circle positions
    circles = circlify.circlify(
        df_p[absolute_values_col_name].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: np.cos(row["angle"]), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: np.sin(row["angle"]), axis=1 
    )
    df_p["x1"] = df_p["center_x"]+df_p["radius"]*df_p["cos_angle"]
    df_p["y1"] = df_p["center_y"]+df_p["radius"]*df_p["sin_angle"]

    cds_p = ColumnDataSource(df_p)
    p.circle(x="center_x", y="center_y", radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="center_y", x1="x1", y1="y1", color=color_col_name, line_width=10, source=cds_p)
    return p


def glyph_keyhole_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p = df_p.sort_values(by=absolute_values_col_name)

    # compute circle positions
    circles = circlify.circlify(
        df_p[absolute_values_col_name].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    # df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    # df_p["cos_angle"] = df_p.apply(
    #     lambda row: np.cos(row["angle"]), axis=1 
    # )
    # df_p["sin_angle"] = df_p.apply(
    #     lambda row: np.sin(row["angle"]), axis=1 
    # )
    # df_p["x1"] = df_p["center_x"]+df_p["radius"]*df_p["cos_angle"]
    # df_p["y1"] = df_p["center_y"]+df_p["radius"]*df_p["sin_angle"]
    df_p["y0"] = df_p["center_y"]-df_p["radius"].min()
    df_p["y1"] = df_p["center_y"]+df_p["radius"].min()
    df_p["y_part"] = df_p["y0"]+(df_p[absolute_values_col_name]/df_p[total_values_col_name])*(df_p["y1"]-df_p["y0"])

    cds_p = ColumnDataSource(df_p)
    p.circle(x="center_x", y="center_y", radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="y0", x1="center_x", y1="y1", color="white", line_width=10, source=cds_p)
    # p.segment(x0="center_x", y0="y0", x1="center_x", y1="y1", color=color_col_name, line_width=10, alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="y0", x1="center_x", y1="y_part", color=color_col_name, line_width=10, source=cds_p)
    return p


def circumference_point_aligned_keyhole_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500) #, x_range=(-1.2,1.2), y_range=(-1.2,1.2)
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    df_p["center_x"] = (df_p["radius"]-df_p["radius"].max())/np.sqrt(2)
    df_p["center_y"] = (df_p["radius"]-df_p["radius"].max())/np.sqrt(2)
    
    # df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    # df_p["cos_angle"] = df_p.apply(
    #     lambda row: np.cos(row["angle"]), axis=1 
    # )
    # df_p["sin_angle"] = df_p.apply(
    #     lambda row: np.sin(row["angle"]), axis=1 
    # )
    # df_p["x1"] = df_p["center_x"]+df_p["radius"]*df_p["cos_angle"]
    # df_p["y1"] = df_p["center_y"]+df_p["radius"]*df_p["sin_angle"]
    df_p["y0"] = df_p["center_y"]-df_p["radius"].min()
    df_p["y1"] = df_p["center_y"]+df_p["radius"].min()
    df_p["y_part"] = df_p["y0"]+(df_p[absolute_values_col_name]/df_p[total_values_col_name])*(df_p["y1"]-df_p["y0"])

    cds_p = ColumnDataSource(df_p)
    p.circle(x="center_x", y="center_y", radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="y0", x1="center_x", y1="y1", color="white", line_width=10, source=cds_p)
    # p.segment(x0="center_x", y0="y0", x1="center_x", y1="y1", color=color_col_name, line_width=10, alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="y0", x1="center_x", y1="y_part", color=color_col_name, line_width=10, source=cds_p)
    return p



def overlapping_keyhole_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    df_p = deepcopy(df)
    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    y_max = df_p["radius"].max()*2

    p = figure(title=None, width=500, height=500, x_range=(-y_max*0.6,y_max*0.6), y_range=(0,y_max*1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None


    df_p["center_y"] = df_p["radius"]
    df_p["y_part"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*y_max

    cds_p = ColumnDataSource(df_p)
    p.circle(x=0, y="center_y", radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.segment(x0=0, y0=0, x1=0, y1=y_max, color="white", line_width=20)
    p.scatter(x=0, y=[i*0.1*y_max for i in range(1,10)], color="black", size=20, marker="dash")
    p.scatter(x=0, y="y_part", color=color_col_name, size=20, alpha=0.7, source=cds_p)
    return p


def bubbles_1d_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):

    df_p = deepcopy(df)
    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    df_p["center_x"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])
    radius_factor = min(df_p["center_x"].min()/df_p["radius"].min(), (1-df_p["center_x"].max())/df_p["radius"].max())*0.3
    df_p["radius"] = df_p["radius"]*radius_factor

    p = figure(title=None, width=500, height=500, x_range=(-0.001,1.001))#, y_range=(0,y_max*1.2)
    # p.grid.visible = False
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None


    cds_p = ColumnDataSource(df_p)
    p.segment(x0=0, y0=0, x1=1, y1=0, color="black")
    p.scatter(x=[i*0.1 for i in range(0,11)], y=0, color="black", size=10, marker="cross")
    p.circle(x="center_x", y=0, radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.scatter(x="center_x", y=0, color=color_col_name, size=5, source=cds_p)
    return p


def circumference_point_aligned_bubble_gauge_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500)
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    df_p["center_x"] = 0
    df_p["center_y"] = df_p["radius"]

    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: np.cos(row["angle"]), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: np.sin(row["angle"]), axis=1 
    )
    df_p["x1"] = df_p["center_x"]+df_p["radius"]*df_p["cos_angle"]
    df_p["y1"] = df_p["center_y"]+df_p["radius"]*df_p["sin_angle"]

    cds_p = ColumnDataSource(df_p)
    p.circle(x="center_x", y="center_y", radius="radius", color=color_col_name, fill_alpha=0.3, source=cds_p)
    p.segment(x0="center_x", y0="center_y", x1="x1", y1="y1", color=color_col_name, line_width=10, source=cds_p)
    return p


def glyph_inverted_T_bar_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    # df_p = df_p.sort_values(by=absolute_values_col_name)

    # compute circle positions
    circles = circlify.circlify(
        [1]*len(df_p.index),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    df_p["x_bottom_left"] = df_p["center_x"] - df_p["radius"] * np.sqrt(3) / 2
    df_p["x_bottom_right"] = df_p["center_x"] + df_p["radius"] * np.sqrt(3) / 2
    df_p["x_top"] = df_p["center_x"]
    df_p["y_bottom"] = df_p["center_y"] - df_p["radius"] / 2
    df_p["x_part"] = df_p["x_bottom_left"] + (df_p[absolute_values_col_name]/df_p[total_values_col_name]) * (df_p["x_bottom_right"] - df_p["x_bottom_left"])
    df_p["y_top"] = df_p["y_bottom"] + (df_p[absolute_values_col_name]/df_p[absolute_values_col_name].max()) * (df_p["radius"]*1.5)

    cds_p = ColumnDataSource(df_p)
    p.segment(x0="x_bottom_left", y0="y_bottom", x1="x_bottom_right", y1="y_bottom", color=color_col_name, line_width=10, alpha=0.3, source=cds_p)
    p.segment(x0="x_bottom_left", y0="y_bottom", x1="x_part", y1="y_bottom", color=color_col_name, line_width=10, source=cds_p)
    p.segment(x0="x_top", y0="y_bottom", x1="x_top", y1="y_top", color=color_col_name, line_width=10, source=cds_p)

    return p


def overlapping_bubble_gauge_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    b_fractional_ticks: bool = True,
    ):

    grid_line_color = "#e5e5e5"

    df_p = deepcopy(df)

    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    max_radius = df_p["radius"].max()*1.1

    p = figure(title=None, width=500, height=500, x_range=(-1.15*max_radius,1.15*max_radius), y_range=(-1.15*max_radius,1.15*max_radius))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)

    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: np.cos(row["angle"]), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: np.sin(row["angle"]), axis=1 
    )
    df_p["x0"] = df_p["radius"]*df_p["cos_angle"]
    df_p["y0"] = df_p["radius"]*df_p["sin_angle"]
    df_p["x1"] = max_radius * df_p["cos_angle"]
    df_p["y1"] = max_radius * df_p["sin_angle"]
    cds_p = ColumnDataSource(df_p)

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    p.circle(x=0, y=0, radius="radius", color=color_col_name, fill_alpha=0.2, source=cds_p)
    p.segment(x0=0, y0=0, x1="x0", y1="y0", color=color_col_name, line_width=5, source=cds_p)
    p.segment(
        source=cds_p,
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_dash="dashed",
        line_cap='round',
        alpha=0.3,
        line_width=5,
    )
    p.scatter(
        source=cds_p,
        x="x1",
        y="y1",
        color=color_col_name,
        size=10,
    )
    return p


def pendulum_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    float_radius_scale: float = 0.38,
    b_fractional_ticks: bool = True,
    ):

    grid_line_color = "#e5e5e5"

    df_p = deepcopy(df)

    df_p["radius"] = np.sqrt(df_p[absolute_values_col_name])
    max_radius = df_p["radius"].max()*(1.1+float_radius_scale)

    p = figure(title=None, width=500, height=500, x_range=(-1.15*max_radius,1.15*max_radius), y_range=(-1.15*max_radius,1.15*max_radius))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)

    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: np.cos(row["angle"]), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: np.sin(row["angle"]), axis=1 
    )
    df_p["x1"] = max_radius * df_p["cos_angle"]
    df_p["y1"] = max_radius * df_p["sin_angle"]
    df_p["radius_small"] = df_p["radius"]*float_radius_scale

    cds_p = ColumnDataSource(df_p)

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    # p.circle(x=0, y=0, radius="radius", color=color_col_name, fill_alpha=0.2, source=cds_p)
    # p.segment(x0=0, y0=0, x1="x0", y1="y0", color=color_col_name, line_width=5, source=cds_p)
    p.segment(
        source=cds_p,
        x0=0,
        y0=0,
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_width=5,
    )
    p.circle(x="x1", y="y1", radius="radius_small", color=color_col_name, fill_alpha=0.2, source=cds_p)

    # p.scatter(
    #     source=cds_p,
    #     x="x1",
    #     y="y1",
    #     color=color_col_name,
    #     size=10,
    # )
    return p


def glyph_donut_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p[total_values_col_name+"_squared"]=df_p[total_values_col_name]*df_p[total_values_col_name]
    df_p = df_p.sort_values(by=total_values_col_name+"_squared")

    # compute circle positions
    circles = circlify.circlify(
        df_p[total_values_col_name+"_squared"].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    # reverse the order of the circles to match the order of data
    # circles = circles[::-1]


    df_p["center_x"] = [x for (x, _, _) in circles]
    df_p["center_y"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.85 for (_, _, r) in circles]
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    cds_p = ColumnDataSource(df_p)
    p.arc(x="center_x", y="center_y", radius="radius", start_angle = np.pi/2, end_angle = np.pi/2+2*np.pi, color=color_col_name, alpha=0.3, line_width=15, source=cds_p)
    p.arc(x="center_x", y="center_y", radius="radius", start_angle = np.pi/2, end_angle = "angle", color=color_col_name, direction="clock", line_width=15, source=cds_p)
    return p


def glyph_wand_plus_charts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(-1.2,1.2), y_range=(-1.2,1.2))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p[absolute_values_col_name+"_squared"]=df_p[absolute_values_col_name]*df_p[absolute_values_col_name]
    df_p = df_p.sort_values(by=absolute_values_col_name+"_squared")

    # compute circle positions
    circles = circlify.circlify(
        df_p[absolute_values_col_name+"_squared"].tolist(),
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    # reverse the order of the circles to match the order of data
    # circles = circles[::-1]


    df_p["x0"] = [x for (x, _, _) in circles]
    df_p["y0"] = [y for (_, y, _) in circles]
    df_p["radius"] = [r*0.9 for (_, _, r) in circles]
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["x1"] = df_p.apply(lambda row: row["x0"] + row["radius"]*np.cos(row["angle"]), axis=1)
    df_p["y1"] = df_p.apply(lambda row: row["y0"] + row["radius"]*np.sin(row["angle"]), axis=1)
    cds_p = ColumnDataSource(df_p)
    # p.arc(x="center_x", y="center_y", radius="radius", start_angle = np.pi/2, end_angle = np.pi/2+2*np.pi, color=color_col_name, alpha=0.3, line_width=15, source=cds_p)
    p.scatter(x="x0", y="y0", color=color_col_name, marker="+", size=50, source=cds_p)
    p.segment(x0="x0", y0="y0", x1="x1", y1="y1", color=color_col_name, line_width=5, source=cds_p)
    return p


def two_level_treemap(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    width = 500
    height = 500
    p = figure(title=None, width=width, height=height, x_range=(0,width), y_range=(0,height))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_two_level_treemap = deepcopy(df)
    df_two_level_treemap = df_two_level_treemap.sort_values(by=total_values_col_name, ascending=False)
    values = list(df_two_level_treemap[total_values_col_name])
    # values must be sorted descending (and positive, obviously)
    values.sort(reverse=True)
    # the sum of the values must equal the total area to be laid out
    # i.e., sum(values) == width * height
    values = squarify.normalize_sizes(values, width, height)
    # returns a list of rectangles
    rects = squarify.squarify(values, 0, 0, width, height)

    df_two_level_treemap["left"] = [rect["x"] for rect in rects]
    df_two_level_treemap["bottom"] = [rect["y"] for rect in rects]
    df_two_level_treemap["right"] = [rect["x"]+rect["dx"] for rect in rects]
    df_two_level_treemap["top"] = [rect["y"]+rect["dy"] for rect in rects]
    df_two_level_treemap["part_right"] = df_two_level_treemap.apply(
        lambda row: (row["left"] + (row["right"]-row["left"])*(row[absolute_values_col_name]/row[total_values_col_name])) 
        if (row["right"]-row["left"])>(row["top"]-row["bottom"]) 
        else row["right"], 
        axis=1
        )
    df_two_level_treemap["part_top"] = df_two_level_treemap.apply(
        lambda row: (row["bottom"] + (row["top"]-row["bottom"])*(row[absolute_values_col_name]/row[total_values_col_name])) 
        if (row["right"]-row["left"])<=(row["top"]-row["bottom"]) 
        else row["top"], 
        axis=1
        )
    cds_two_level_treemap = ColumnDataSource(df_two_level_treemap)
    p.quad(left="left", bottom="bottom", right="right", top="top", color=color_col_name, alpha=0.3, source=cds_two_level_treemap)
    p.quad(left="left", bottom="bottom", right="part_right", top="part_top", color=color_col_name, source=cds_two_level_treemap)
    return p


def two_level_icicle_plot(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    width = 500
    height = 500

    sum_y = df[total_values_col_name].sum()
    max_y = sum_y

    p = figure(title=None, width=width, height=height, x_range=(0,1), y_range=(-max_y*0.1,max_y*1.1))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_two_level_icicle_plot = deepcopy(df)
    df_two_level_icicle_plot["top"] = df_two_level_icicle_plot[total_values_col_name].cumsum()
    df_two_level_icicle_plot["bottom"] = df_two_level_icicle_plot["top"] - df_two_level_icicle_plot[total_values_col_name]
    df_two_level_icicle_plot["part_top"] = df_two_level_icicle_plot["bottom"] + df_two_level_icicle_plot[absolute_values_col_name]
    cds_two_level_icicle_plot = ColumnDataSource(df_two_level_icicle_plot)
    p.quad(left=0.3, bottom="bottom", right=0.5, top="top", color=color_col_name, fill_alpha=0.1, hatch_pattern="|", hatch_scale=4.0, source=cds_two_level_icicle_plot)
    p.quad(left=0.5, bottom="bottom", right=0.7, top="part_top", color=color_col_name, source=cds_two_level_icicle_plot)
    p.quad(left=0.5, bottom="part_top", right=0.7, top="top", color=color_col_name, alpha=0.3, source=cds_two_level_icicle_plot)


    return p


def two_level_parallel_set_plot(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    width = 500
    height = 500

    sum_y = df[total_values_col_name].sum()
    padding = sum_y*0.2
    max_y = sum_y+padding

    p = figure(title=None, width=width, height=height, x_range=(-0.25,1.25), y_range=(-max_y*0.1,max_y*1.1))
    p.grid.visible = False
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_parallel_set_plot_lvl1 = deepcopy(df)
    df_parallel_set_plot_lvl1["padding"] = [(i*padding/(len(df.index)-1) if len(df.index) > 1 else 0) for i in list(range(0, len(df.index)))]
    df_parallel_set_plot_lvl1["top"] = df_parallel_set_plot_lvl1[total_values_col_name].cumsum() + df_parallel_set_plot_lvl1["padding"]
    df_parallel_set_plot_lvl1["bottom"] = df_parallel_set_plot_lvl1["top"] - df_parallel_set_plot_lvl1[total_values_col_name]
    df_parallel_set_plot_lvl1["part_top"] = df_parallel_set_plot_lvl1["bottom"] + df_parallel_set_plot_lvl1[absolute_values_col_name]
    cds_parallel_set_plot_lvl1 = ColumnDataSource(df_parallel_set_plot_lvl1)
    p.quad(left=-0.1, bottom="bottom", right=0.1, top="top", color=color_col_name, alpha=0.3, source=cds_parallel_set_plot_lvl1)
    p.quad(left=-0.1, bottom="bottom", right=0.1, top="part_top", color=color_col_name, source=cds_parallel_set_plot_lvl1)

    df_parallel_set_plot_lvl2_bottom = deepcopy(df)
    df_parallel_set_plot_lvl2_bottom["top"] = df_parallel_set_plot_lvl2_bottom[absolute_values_col_name].cumsum()
    df_parallel_set_plot_lvl2_bottom["bottom"] = df_parallel_set_plot_lvl2_bottom["top"] - df_parallel_set_plot_lvl2_bottom[absolute_values_col_name]
    cds_parallel_set_plot_lvl2_bottom = ColumnDataSource(df_parallel_set_plot_lvl2_bottom)
    p.quad(left=0.9, bottom="bottom", right=1.1, top="top", color=color_col_name, source=cds_parallel_set_plot_lvl2_bottom)

    df_parallel_set_plot_lvl2_top = deepcopy(df)
    df_parallel_set_plot_lvl2_top["difference"] = df_parallel_set_plot_lvl2_top[total_values_col_name] - df_parallel_set_plot_lvl2_top[absolute_values_col_name]
    df_parallel_set_plot_lvl2_top["top"] = df_parallel_set_plot_lvl2_top["difference"].cumsum() + padding + df[absolute_values_col_name].sum()
    df_parallel_set_plot_lvl2_top["bottom"] = df_parallel_set_plot_lvl2_top["top"] - df_parallel_set_plot_lvl2_top["difference"]
    cds_parallel_set_plot_lvl2_top = ColumnDataSource(df_parallel_set_plot_lvl2_top)
    p.quad(left=0.9, bottom="bottom", right=1.1, top="top", color=color_col_name, alpha=0.3, source=cds_parallel_set_plot_lvl2_top)

    df_parallel_set_plot_edges = deepcopy(df)
    df_parallel_set_plot_edges["xs"]=[[0.1, 0.1, 0.9, 0.9]]*len(df.index)
    df_parallel_set_plot_edges["bottom_left"] = df_parallel_set_plot_lvl1["bottom"]
    df_parallel_set_plot_edges["top_left"] = df_parallel_set_plot_lvl1["part_top"]
    df_parallel_set_plot_edges["top_right"] = df_parallel_set_plot_lvl2_bottom["top"]
    df_parallel_set_plot_edges["bottom_right"] = df_parallel_set_plot_lvl2_bottom["bottom"]
    df_parallel_set_plot_edges["ys"]=df_parallel_set_plot_edges.apply(lambda row: [row["bottom_left"], row["top_left"], row["top_right"], row["bottom_right"],], axis = 1)
    cds_parallel_set_plot_edges = ColumnDataSource(df_parallel_set_plot_edges)
    p.patches(xs="xs", ys="ys", color=color_col_name, fill_alpha=0.1, hatch_pattern ="+", hatch_scale = 3, source=cds_parallel_set_plot_edges)

    return p


def marimekko_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(0,df[total_values_col_name].sum()), y_range=(0,1))
    p.xgrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_marimekko_chart = deepcopy(df)
    df_marimekko_chart["right"] = df_marimekko_chart[total_values_col_name].cumsum()
    df_marimekko_chart["left"] = df_marimekko_chart["right"] - df_marimekko_chart[total_values_col_name]
    df_marimekko_chart["top"] = df_marimekko_chart[absolute_values_col_name]/df_marimekko_chart[total_values_col_name]
    cds_marimekko_chart = ColumnDataSource(df_marimekko_chart)
    p.quad(left="left", bottom=0, right="right", top=1, color=color_col_name, alpha=0.3, source=cds_marimekko_chart)
    p.quad(left="left", bottom=0, right="right", top="top", color=color_col_name, source=cds_marimekko_chart)

    return p


def inverted_triangle_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    padding = df[total_values_col_name].sum()*0.2
    p = figure(title=None, width=500, height=500, x_range=(0,df[total_values_col_name].sum()+padding), y_range=(0,1))
    p.xgrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_inverted_triangle_chart = deepcopy(df)
    df_inverted_triangle_chart["padding"] = [i*padding/(len(df.index) + 1) for i in list(range(1, len(df.index)+1))]
    df_inverted_triangle_chart["right"] = df_inverted_triangle_chart[total_values_col_name].cumsum() + df_inverted_triangle_chart["padding"]
    df_inverted_triangle_chart["left"] = df_inverted_triangle_chart["right"] - df_inverted_triangle_chart[total_values_col_name]
    df_inverted_triangle_chart["center"] = df_inverted_triangle_chart["left"] + df_inverted_triangle_chart[total_values_col_name]/2
    df_inverted_triangle_chart["top"] = df_inverted_triangle_chart[absolute_values_col_name]/df_inverted_triangle_chart[total_values_col_name]
    df_inverted_triangle_chart["xs"] = df_inverted_triangle_chart.apply(lambda row: [row["left"], row["right"], row["center"]], axis=1)
    df_inverted_triangle_chart["ys"] = df_inverted_triangle_chart.apply(lambda row: [row["top"], row["top"], 0], axis=1)
    cds_inverted_triangle_chart = ColumnDataSource(df_inverted_triangle_chart)
    p.patches(xs="xs", ys="ys", color=color_col_name, source=cds_inverted_triangle_chart)

    return p


def inverted_triangle_chart_overlapping(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    ):
    p = figure(title=None, width=500, height=500, x_range=(0,df[total_values_col_name].max()), y_range=(0,1))
    p.xgrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_inverted_triangle_chart = deepcopy(df)
    df_inverted_triangle_chart["right"] = df_inverted_triangle_chart[total_values_col_name]
    df_inverted_triangle_chart["top"] = df_inverted_triangle_chart[absolute_values_col_name]/df_inverted_triangle_chart[total_values_col_name]
    df_inverted_triangle_chart["xs"] = df_inverted_triangle_chart.apply(lambda row: [0, row["right"], row["right"]], axis=1)
    df_inverted_triangle_chart["ys"] = df_inverted_triangle_chart.apply(lambda row: [row["top"], row["top"], 0], axis=1)
    cds_inverted_triangle_chart = ColumnDataSource(df_inverted_triangle_chart)
    p.patches(xs="xs", ys="ys", color=color_col_name, source=cds_inverted_triangle_chart)

    return p


def radial_bar_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    fractional_values_radial_grid_lines_visibility: bool = True,
    absolute_values_circular_grid_lines_visibility: bool = True,
    b_fractional_ticks: bool = True,
):
    padding = 0.2
    major_tick_interval = pow(10,np.ceil(np.log10(df[absolute_values_col_name].max()))-2)
    grid_line_color = "#e5e5e5"

    #
    # Generating the figure
    #
    max_radius = df[absolute_values_col_name].max() * (1 + padding)
    max_radius_plot = max_radius * 1.25
    p = figure(
        height=500,
        width=500,
        x_range=(-max_radius_plot, max_radius_plot),
        y_range=(-max_radius_plot, max_radius_plot),
    )
    p.yaxis.fixed_location = 0
    p.yaxis.bounds = (0, max_radius)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)

    #
    # Drawing grid lines
    #
    if fractional_values_radial_grid_lines_visibility:
        p.segment(
            x0=0,
            y0=0,
            x1=[max_radius * 1.02 * np.cos(theta) for theta in [0, -np.pi/2, -np.pi]],
            y1=[max_radius * 1.02 * np.sin(theta) for theta in [0, -np.pi/2, -np.pi]],
            color=grid_line_color,
            level="underlay",
        )
    if absolute_values_circular_grid_lines_visibility:
        x_major_ticks_values = np.arange(
            major_tick_interval*5, max_radius, major_tick_interval*5
        )
        p.ellipse(
            x=0,
            y=0,
            width=x_major_ticks_values * 2,
            height=x_major_ticks_values * 2,
            line_color=grid_line_color,
            fill_color=None,
            level="underlay",
        )

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    df_p = deepcopy(df)
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: round(np.cos(row["angle"]),2), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: round(np.sin(row["angle"]),2), axis=1 
    )
    df_p["x0"] = df_p[absolute_values_col_name] * df_p["cos_angle"]
    df_p["y0"] = df_p[absolute_values_col_name] * df_p["sin_angle"]
    df_p["x1"] = (max_radius * 1.01) * df_p["cos_angle"]
    df_p["y1"] = (max_radius * 1.01) * df_p["sin_angle"]
    cds_p = ColumnDataSource(df_p)

    p.segment(
        source=cds_p,
        x0=0,
        y0=0,
        x1="x0",
        y1="y0",
        line_color=color_col_name,
        line_width=5,
    )

    p.segment(
        source=cds_p,
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_dash="dashed",
        line_cap='round',
        alpha=0.3,
        line_width=2,
    )

    p.scatter(
        source=cds_p,
        x="x1",
        y="y1",
        color=color_col_name,
        size=10,
    )

    return p


def radial_bar_chart_with_absolute_values_intercepts(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    fractional_values_radial_grid_lines_visibility: bool = True,
    absolute_values_circular_grid_lines_visibility: bool = False,
    b_fractional_ticks: bool = True,
):
    padding = 0.2
    major_tick_interval = pow(10,np.ceil(np.log10(df[absolute_values_col_name].max()))-2)
    grid_line_color = "#e5e5e5"

    #
    # Generating the figure
    #
    max_radius = df[absolute_values_col_name].max() * (1 + padding)
    max_radius_plot = max_radius * 1.25
    p = figure(
        height=500,
        width=500,
        x_range=(-max_radius_plot, max_radius_plot),
        y_range=(-max_radius_plot, max_radius_plot),
    )
    p.yaxis.fixed_location = 0
    p.yaxis.bounds = (0, max_radius)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)

    #
    # Drawing grid lines
    #
    if fractional_values_radial_grid_lines_visibility:
        p.segment(
            x0=0,
            y0=0,
            x1=[max_radius * 1.02 * np.cos(theta) for theta in [0, -np.pi/2, -np.pi]],
            y1=[max_radius * 1.02 * np.sin(theta) for theta in [0, -np.pi/2, -np.pi]],
            color=grid_line_color,
            level="underlay",
        )
    if absolute_values_circular_grid_lines_visibility:
        x_major_ticks_values = np.arange(
            major_tick_interval*5, max_radius, major_tick_interval*5
        )
        p.ellipse(
            x=0,
            y=0,
            width=x_major_ticks_values * 2,
            height=x_major_ticks_values * 2,
            line_color=grid_line_color,
            fill_color=None,
            level="underlay",
        )

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    df_p = deepcopy(df)
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: round(np.cos(row["angle"]),2), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: round(np.sin(row["angle"]),2), axis=1 
    )
    df_p["x0"] = df_p[absolute_values_col_name] * df_p["cos_angle"]
    df_p["y0"] = df_p[absolute_values_col_name] * df_p["sin_angle"]
    df_p["x1"] = (max_radius * 1.01) * df_p["cos_angle"]
    df_p["y1"] = (max_radius * 1.01) * df_p["sin_angle"]
    cds_p = ColumnDataSource(df_p)

    p.segment(
        source=cds_p,
        x0=0,
        y0=0,
        x1="x0",
        y1="y0",
        line_color=color_col_name,
        line_width=5,
    )

    p.segment(
        source=cds_p,
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_dash="dashed",
        line_cap='round',
        alpha=0.3,
        line_width=1,
    )

    p.scatter(
        source=cds_p,
        x="x1",
        y="y1",
        color=color_col_name,
        size=5,
    )

    p.arc(
        x=0, 
        y=0, 
        radius=absolute_values_col_name, 
        start_angle = np.pi/2, 
        end_angle = "angle", 
        color=color_col_name,
        direction="clock", 
        line_dash="dashed", 
        # alpha=0.3, 
        line_width=2, 
        source=cds_p
        )

    p.scatter(
        source=cds_p,
        x=0,
        y=absolute_values_col_name,
        color=color_col_name,
        size=20,
        level="overlay"
    )


    return p


def overlapping_pie_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    fractional_values_radial_grid_lines_visibility: bool = True,
    b_fractional_ticks: bool = True,
):
    padding = 0.2
    grid_line_color = "#e5e5e5"

    #
    # Generating the figure
    #
    max_total_value = 1
    major_tick_interval = pow(10,np.ceil(np.log10(max_total_value))-2)
    max_radius = max_total_value * (1 + padding)
    max_radius_plot = max_radius * 1.25
    p = figure(
        height=500,
        width=500,
        x_range=(-max_radius_plot, max_radius_plot),
        y_range=(-max_radius_plot, max_radius_plot),
    )
    p.yaxis.visible = False
    p.xaxis.fixed_location = 0
    p.xaxis.bounds = (0, max_radius)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    p.segment(
        x0=0,
        y0=0,
        x1=max_radius * 1.02 * np.cos(np.pi/2),
        y1=max_radius * 1.02 * np.sin(np.pi/2),
        color="black",
        level="underlay",
    )

    if b_fractional_ticks:
        # tick for every 5 percent
        angular_ticks_values = np.arange(np.pi/2, np.pi * 2.5, np.pi / 10)
        p.segment(
            x0=[max_radius * np.cos(theta) for theta in angular_ticks_values],
            y0=[max_radius * np.sin(theta) for theta in angular_ticks_values],
            x1=[max_radius * np.cos(theta) * 1.03 for theta in angular_ticks_values],
            y1=[max_radius * np.sin(theta) * 1.03 for theta in angular_ticks_values],
            color="black",
            level="underlay",
        )
        angle_tick_labels_cds = ColumnDataSource(
            data=dict(
                x=[max_radius * 1.04 * np.cos(theta) for theta in angular_ticks_values],
                y=[max_radius * 1.04 * np.sin(theta) for theta in angular_ticks_values],
                angle=angular_ticks_values-np.pi/2,
                text=[
                    "{:.0f}%".format(125-(round(value,2)*100))
                    for value in (angular_ticks_values * 0.5 / np.pi)
                ],
            )
        )
        angle_tick_labels = LabelSet(
            source=angle_tick_labels_cds,
            x="x",
            y="y",
            text="text",
            angle="angle",
            text_align='center',
            text_font_size = "10pt",
            x_offset=0,
            y_offset=0,
            # render_mode="canvas",
        )
        p.add_layout(angle_tick_labels)



    #
    # Drawing grid lines
    #
    if fractional_values_radial_grid_lines_visibility:
        p.segment(
            x0=0,
            y0=0,
            x1=[max_radius * 1.02 * np.cos(theta) for theta in [0, -np.pi/2, -np.pi]],
            y1=[max_radius * 1.02 * np.sin(theta) for theta in [0, -np.pi/2, -np.pi]],
            color=grid_line_color,
            level="underlay",
        )

    #
    # Outer ring
    #
    p.ellipse(
        x=0,
        y=0,
        width=[max_radius * 2, max_radius * 1.02 * 2],
        height=[max_radius * 2, max_radius * 1.02 * 2],
        line_color=grid_line_color,
        fill_color=None,
        level="underlay",
    )

    df_p = deepcopy(df)
    df_p["radius"] = df_p.apply(
        lambda row: np.sqrt(row[total_values_col_name])/np.sqrt(df[total_values_col_name].max()), axis=1
        )
    df_p["angle"] = (df_p[absolute_values_col_name]/df_p[total_values_col_name])*(-2 * np.pi) + np.pi*0.5
    df_p["cos_angle"] = df_p.apply(
        lambda row: round(np.cos(row["angle"]),2), axis=1 
    )
    df_p["sin_angle"] = df_p.apply(
        lambda row: round(np.sin(row["angle"]),2), axis=1 
    )
    df_p["x0"] = df_p["radius"] * df_p["cos_angle"]
    df_p["y0"] = df_p["radius"] * df_p["sin_angle"]
    df_p["x1"] = (max_radius * 1.01) * df_p["cos_angle"]
    df_p["y1"] = (max_radius * 1.01) * df_p["sin_angle"]
    cds_p = ColumnDataSource(df_p)

    p.circle(x=0, y=0, radius="radius", color=color_col_name, fill_alpha=0.1, source=cds_p)

    p.segment(
        source=cds_p,
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color=color_col_name,
        line_dash="dashed",
        line_cap='round',
        alpha=0.3,
        line_width=2,
    )

    p.scatter(
        source=cds_p,
        x="x1",
        y="y1",
        color=color_col_name,
        size=10,
    )

    angle_list = (np.arange(0, 1.00001, 1/(100)))*(-2 * np.pi) + np.pi*0.5
    df_pattern=pd.DataFrame({"start_angle":angle_list[:-1], "end_angle":angle_list[1:]})
    df_pattern["radius"] = repeat_list(list(df_p["radius"]), len(df_pattern.index))
    df_pattern[color_col_name] = repeat_list(list(df_p[color_col_name]), len(df_pattern.index))
    df_pattern["angle_orig"] = repeat_list(list(df_p["angle"]), len(df_pattern.index))
    df_pattern["end_angle"] = df_pattern.apply(
        lambda row: row["angle_orig"] 
        if (row["end_angle"] < row["angle_orig"] and row["start_angle"] > row["angle_orig"]) 
        else row["end_angle"],
        axis=1)
    df_pattern["alpha"] = df_pattern.apply(lambda row: 0.0 if row["end_angle"] < row["angle_orig"] else 0.8,axis=1)
    cds_pattern = ColumnDataSource(df_pattern)
    p.wedge(x=0, y=0, radius="radius", start_angle = "start_angle", end_angle = "end_angle", alpha = "alpha", color=color_col_name,  direction="clock", source=cds_pattern)

    p.wedge(x=0, y=0, radius="radius", start_angle = np.pi/2, end_angle = "angle", fill_alpha = 0, color=color_col_name, direction="clock", source=cds_p)


    return p


def overlapping_circular_area_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
    b_fractional_ticks: bool = True,
):
    padding = 0.2
    grid_line_color = "#e5e5e5"

    #
    # Generating the figure
    #
    max_total_value = 1
    max_radius = max_total_value * (1 + padding)
    max_radius_plot = max_radius * 1.25
    p = figure(
        height=500,
        width=500,
        x_range=(-max_radius_plot, max_radius_plot),
        y_range=(-max_radius_plot, max_radius_plot),
    )
    p.yaxis.visible = False
    p.xaxis.fixed_location = 0
    p.xaxis.bounds = (0, max_radius)
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    df_p = deepcopy(df)
    df_p["radius"] = df_p.apply(
        lambda row: np.sqrt(row[total_values_col_name])/np.sqrt(df[total_values_col_name].max()), axis=1
        )
    df_p["part_radius"] = df_p.apply(
        lambda row: np.sqrt(row[absolute_values_col_name])/np.sqrt(df[total_values_col_name].max()), axis=1
        )
    cds_p = ColumnDataSource(df_p)
    p.circle(x=0, y=0, radius="radius", color=color_col_name, fill_alpha=0.1, line_width=2, source=cds_p)
    p.circle(x=0, y=0, radius="part_radius", color=color_col_name, fill_alpha=0, source=cds_p)

    angle_list = (np.arange(0, 1.00001, 1/(100-100%len(df.index))))*(-2 * np.pi) + np.pi*0.5
    df_pattern=pd.DataFrame({"start_angle":angle_list[:-1], "end_angle":angle_list[1:]})
    df_pattern["radius"] = repeat_list(list(df_p["part_radius"]), len(df_pattern.index))
    df_pattern[color_col_name] = repeat_list(list(df_p[color_col_name]), len(df_pattern.index))
    cds_pattern = ColumnDataSource(df_pattern)
    p.wedge(x=0, y=0, radius="radius", start_angle = "start_angle", end_angle = "end_angle", color=color_col_name,  direction="clock", source=cds_pattern)

    return p


def spring_bar_chart(
    df: pd.DataFrame,
    absolute_values_col_name: str,
    total_values_col_name: str,
    id_col_name: str,
    color_col_name: str,
):

    #
    # Generating the figure
    #

    p = figure(
        height=500,
        width=500,
        x_range=(-1/(2*len(df.index)), 1-1/(2*len(df.index))),
        # y_range=(0, 1+1/(len(df.index))),
        y_range=(0, 1.04),
        # match_aspect=True
    )
    p.xgrid.visible = False
    p.xaxis.visible = False
    p.toolbar.logo = None
    p.toolbar_location = None

    width_spring_pan = 0.8/(2*len(df.index)) # 
    max_width = 0.7*width_spring_pan # max width of spring
    major_tick_interval = pow(10,np.ceil(np.log10(df[total_values_col_name].max()))-2)/2
    while max_width**2 + ((major_tick_interval/df[total_values_col_name].max())**2 - (major_tick_interval/df[total_values_col_name].min())**2)<=0: 
        major_tick_interval /= 10

    df_bars = deepcopy(df)
    df_bars["center"] = df_bars.index/(len(df.index))
    df_bars["left"] = df_bars["center"]-width_spring_pan
    df_bars["right"] = df_bars["center"]+width_spring_pan
    df_bars["top"] = df_bars[absolute_values_col_name]/df_bars[total_values_col_name]
    df_bars["tick_height"] = major_tick_interval/df_bars[total_values_col_name]
    h_square = max_width**2+(df_bars["tick_height"].min())**2
    df_bars["tick_width"] = df_bars.apply(lambda row: np.sqrt(h_square - row["tick_height"]**2), axis=1)
    df_bars["ys"] = df_bars.apply(lambda row: list(np.arange(0,row[absolute_values_col_name]/row[total_values_col_name]+row["tick_height"]*0.00000001,row["tick_height"])), axis=1)
    df_bars["xs"] = df_bars.apply(
        lambda row: repeat_list(
            [row["center"], row["center"]-row["tick_width"], row["center"], row["center"]+row["tick_width"]], 
            len(row["ys"])
            ), 
            axis=1)
    df_bars["ys"] = df_bars.apply(lambda row: row["ys"]+[(row[absolute_values_col_name]/row[total_values_col_name])], axis=1)
    df_bars["ys_len"] = df_bars.apply(lambda row: len(row["ys"]), axis=1)
    df_bars["xs"] = df_bars.apply(
        lambda row: 
        row["xs"]+
        [spring_bar_chart_width_plus(row, (row["ys"][-1]-row["ys"][-2])*row["tick_width"]/row["tick_height"])],
        axis=1)


    cds_bars = ColumnDataSource(df_bars)
    p.segment(x0="left", y0="top", x1="right", y1="top", color=color_col_name,  line_width=3, source=cds_bars)
    p.multi_line(xs="xs", ys="ys", color=color_col_name,  line_width=3, source=cds_bars)

    hline = Span(location=1, dimension='width', line_color='black', line_dash="dashed", line_width=2)
    p.renderers.extend([hline])

    return p


def spring_bar_chart_width_plus(row, v):
    l = (len(row["ys"])-1)%4
    if l==0:
        return row["center"]+row["tick_width"]-v
    elif l==1:
        return row["center"]-v
    elif l==2:
        return row["center"]-row["tick_width"]+v
    else:
        return row["center"]+v
