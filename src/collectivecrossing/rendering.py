"""Rendering utilities for the collective crossing environment."""

from typing import Any

import numpy as np


def _render_matplotlib(self: Any) -> np.ndarray:
    """Return an RGB array via Agg without touching pyplot (safe for animations)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    # Make a Figure that is NOT connected to any GUI backend
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvas(fig)  # Agg canvas
    ax = fig.add_subplot(1, 1, 1)

    # Draw everything
    self._draw_matplotlib(ax)

    # Avoid pyplot tight_layout; use OO API:
    fig.tight_layout()

    # Render to buffer
    canvas.draw()
    width, height = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
    return arr[..., :3]  # RGB


def draw_matplotlib(env: Any, ax: Any) -> None:
    """
    Draw the environment using matplotlib.

    Args:
    ----
        env: The CollectiveCrossingEnv instance
        ax: The matplotlib axis to draw on

    """
    import matplotlib.patches as patches

    colors = {
        "background": "#f8f9fa",
        "tram_area": "#e3f2fd",
        "waiting_area": "#fff3e0",
        "exiting_destination_area": "#f44336",  # Red color for exit area
        "boarding_destination_area": "#2196f3",  # Blue color for seats area
        "tram_wall": "#424242",
        "door": "#90caf9",  # Light blue, darker than tram area
        "boarding_agent": "#f44336",
        "exiting_agent": "#2196f3",
    }

    ax.set_facecolor(colors["background"])

    # ----- your original drawing code (rects, circles, texts, legend) -----
    # tram area
    ax.add_patch(
        patches.Rectangle(
            (env.tram_left, env.config.division_y),
            env.tram_right - env.tram_left + 1,
            env.config.height - env.config.division_y,
            facecolor=colors["tram_area"],
            edgecolor="none",
            alpha=0.7,
        )
    )

    # waiting area
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            env.config.width,
            env.config.division_y,
            facecolor=colors["waiting_area"],
            edgecolor="none",
            alpha=0.7,
        )
    )

    # exiting destination area
    if env.config.exiting_destination_area_y < env.config.division_y:
        ax.add_patch(
            patches.Rectangle(
                (0, env.config.exiting_destination_area_y),
                env.config.width,
                1,
                facecolor=colors["exiting_destination_area"],
                edgecolor="none",
                alpha=0.8,
            )
        )

    # boarding destination area (limited to tram geometry)
    if env.config.boarding_destination_area_y >= env.config.division_y:
        # If destination equals height, draw seat area below it
        if env.config.boarding_destination_area_y == env.config.height:
            # Draw seat area as a single row below the destination (at height-1)
            ax.add_patch(
                patches.Rectangle(
                    (env.tram_left, env.config.height - 1),
                    env.tram_right - env.tram_left + 1,
                    1,
                    facecolor=colors["boarding_destination_area"],
                    edgecolor="none",
                    alpha=0.8,
                )
            )
        else:
            # Normal case: draw destination area at the specified y-coordinate
            ax.add_patch(
                patches.Rectangle(
                    (env.tram_left, env.config.boarding_destination_area_y),
                    env.tram_right - env.tram_left + 1,
                    1,
                    facecolor=colors["boarding_destination_area"],
                    edgecolor="none",
                    alpha=0.8,
                )
            )

    wall_thickness = 0.1
    # Left wall - extend to cover the left door boundary
    if env.tram_door_left > env.tram_left:
        ax.add_patch(
            patches.Rectangle(
                (env.tram_left, env.config.division_y - wall_thickness / 2),
                env.tram_door_left - env.tram_left + 0.5,  # Extend 0.5 into door area
                wall_thickness,
                facecolor=colors["tram_wall"],
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
            )
        )
    # Right wall - extend to cover the right door boundary
    if env.tram_door_right < env.tram_right:
        ax.add_patch(
            patches.Rectangle(
                (
                    env.tram_door_right - 0.5,
                    env.config.division_y - wall_thickness / 2,
                ),  # Start 0.5 before door boundary
                env.tram_right - env.tram_door_right + 1.5,  # Extend to connect with vertical wall
                wall_thickness,
                facecolor=colors["tram_wall"],
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
            )
        )

    for y in range(env.config.division_y, env.config.height):
        ax.add_patch(
            patches.Rectangle(
                (env.tram_left - wall_thickness / 2, y),
                wall_thickness,
                1,
                facecolor=colors["tram_wall"],
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
            )
        )
        ax.add_patch(
            patches.Rectangle(
                (env.tram_right + 1 - wall_thickness / 2, y),
                wall_thickness,
                1,
                facecolor=colors["tram_wall"],
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
            )
        )

    # Draw door area - only the passable interior positions
    # Door boundaries are exclusive, so only show the interior passable area
    door_width = env.tram_door_right - env.tram_door_left - 1  # Interior width only
    if door_width > 0:  # Only draw if there's a passable interior
        ax.add_patch(
            patches.Rectangle(
                (env.tram_door_left + 0.5, env.config.division_y),
                door_width,
                1,
                facecolor=colors["door"],
                edgecolor="none",
                alpha=0.8,
            )
        )

    # Add text labels directly on the graph
    # Tram area label
    tram_center_x = (env.tram_left + env.tram_right) / 2
    tram_center_y = env.config.division_y + (env.config.height - env.config.division_y) / 2
    ax.text(
        tram_center_x,
        tram_center_y,
        "TRAM",
        fontsize=12,
        weight="bold",
        ha="center",
        va="center",
        color="darkblue",
        alpha=0.8,
    )

    # Waiting area label
    waiting_center_x = env.config.width / 2
    waiting_center_y = env.config.division_y / 2
    ax.text(
        waiting_center_x,
        waiting_center_y,
        "PLATFORM",
        fontsize=12,
        weight="bold",
        ha="center",
        va="center",
        color="darkorange",
        alpha=0.8,
    )

    # Tram door label
    door_center_x = (env.tram_door_left + env.tram_door_right) / 2
    door_center_y = env.config.division_y + 0.5
    ax.text(
        door_center_x,
        door_center_y,
        "DOOR",
        fontsize=10,
        weight="bold",
        ha="center",
        va="center",
        color="darkblue",
        alpha=0.9,
    )

    # Exiting destination area label
    if env.config.exiting_destination_area_y < env.config.division_y:
        exit_center_x = env.config.width / 2
        exit_center_y = env.config.exiting_destination_area_y + 0.5
        ax.text(
            exit_center_x,
            exit_center_y,
            "EXIT",
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            color="white",
            alpha=0.9,
        )

    # Boarding destination area label
    if env.config.boarding_destination_area_y >= env.config.division_y:
        boarding_center_x = (env.tram_left + env.tram_right) / 2
        # If destination equals height, place label at height-1 (where seat area is drawn)
        if env.config.boarding_destination_area_y == env.config.height:
            boarding_center_y = env.config.height - 0.5
        else:
            boarding_center_y = env.config.boarding_destination_area_y + 0.5
        ax.text(
            boarding_center_x,
            boarding_center_y,
            "SEATS",
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            color="white",
            alpha=0.9,
        )

    # Draw all agents using the unified Agent structure
    for agent_id, agent in env._agents.items():
        x, y = agent.x, agent.y

        # Choose color based on agent type
        if agent.is_boarding:
            color = colors["boarding_agent"]
            edge_color = "darkred"
        else:  # exiting
            color = colors["exiting_agent"]
            edge_color = "darkblue"

        # Draw agent with multiple circles for depth effect
        for r, a in [(0.4, 0.3), (0.3, 0.5), (0.2, 0.8)]:
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    r,
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=1,
                    alpha=a,
                )
            )

        # Add agent label
        label = agent_id.split("_", 1)[-1] if "_" in agent_id else agent_id
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
        )

    ax.set_xlim(0, env.config.width)
    ax.set_ylim(0, env.config.height)
    ax.set_aspect("equal")

    # Add prominent grid
    ax.grid(True, alpha=0.7, linestyle="-", linewidth=0.8, color="gray")
    ax.set_xticks(range(env.config.width + 1))
    ax.set_yticks(range(env.config.height + 1))
    ax.tick_params(axis="both", which="major", labelsize=8, colors="darkgray")

    ax.set_title("Collective Crossing Environment", fontsize=14, weight="bold", pad=20)

    # Create legend elements only for agent types (areas are labeled on graph)
    legend_elements = [
        patches.Circle((0, 0), 0.1, facecolor=colors["boarding_agent"], label="Boarding Agents"),
        patches.Circle((0, 0), 0.1, facecolor=colors["exiting_agent"], label="Exiting Agents"),
    ]

    # Place legend below the image
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),  # Below the image
        ncol=2,  # Two columns
        frameon=True,
        fancybox=True,
        shadow=True,
    )
