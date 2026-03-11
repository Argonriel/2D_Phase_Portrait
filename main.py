import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import warnings
import inspect
from datetime import datetime  # 引入时间模块

warnings.filterwarnings('ignore', category=RuntimeWarning)


class PhasePortrait2D:
    def __init__(self, system, x_range="auto", y_range="auto", search_limit=15.0, max_auto_fps=3,
                 title="Phase Portrait"):
        self.system = system
        self.title = title

        try:
            source = inspect.getsource(system)
            dx_line = [line.strip() for line in source.split('\n') if line.strip().startswith('dx =')][0]
            dy_line = [line.strip() for line in source.split('\n') if line.strip().startswith('dy =')][0]
            self.dx_rhs = dx_line.split('=', 1)[1].strip()
            self.dy_rhs = dy_line.split('=', 1)[1].strip()
        except Exception:
            self.dx_rhs = "dx/dt"
            self.dy_rhs = "dy/dt"

        if x_range == "auto" or y_range == "auto":
            auto_x, auto_y = self._auto_determine_range(search_limit, max_auto_fps)
            self.x_range = auto_x if x_range == "auto" else x_range
            self.y_range = auto_y if y_range == "auto" else y_range
        else:
            self.x_range = x_range
            self.y_range = y_range

        self.X, self.Y = np.meshgrid(
            np.linspace(self.x_range[0], self.x_range[1], 400),
            np.linspace(self.y_range[0], self.y_range[1], 400)
        )
        self.U, self.V = self.system(self.X, self.Y)

    def _auto_determine_range(self, limit, max_fps):
        fixed_points = []
        guess_x = np.linspace(-limit, limit, 15)
        guess_y = np.linspace(-limit, limit, 15)

        def root_func(vars):
            return self.system(vars[0], vars[1])

        for gx in guess_x:
            for gy in guess_y:
                sol, _, ier, _ = fsolve(root_func, [gx, gy], full_output=True)
                if ier == 1:
                    if (-limit <= sol[0] <= limit) and (-limit <= sol[1] <= limit):
                        is_new = True
                        for fp in fixed_points:
                            if np.linalg.norm(sol - fp) < 1e-3:
                                is_new = False
                                break
                        if is_new:
                            fixed_points.append(sol)

        if len(fixed_points) == 0:
            return [-5.0, 5.0], [-5.0, 5.0]

        if len(fixed_points) > max_fps:
            fixed_points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
            fixed_points = fixed_points[:max_fps]

        if len(fixed_points) == 1:
            fp = fixed_points[0]
            return [fp[0] - 3, fp[0] + 3], [fp[1] - 3, fp[1] + 3]
        else:
            pts = np.array(fixed_points)
            min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
            min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
            span_x, span_y = max(max_x - min_x, 2.0), max(max_y - min_y, 2.0)
            return [min_x - span_x * 0.3, max_x + span_x * 0.3], [min_y - span_y * 0.3, max_y + span_y * 0.3]

    def _get_jacobian(self, x, y):
        eps = 1e-5
        dx0, dy0 = self.system(x, y)
        dx_x, dy_x = self.system(x + eps, y)
        dx_y, dy_y = self.system(x, y + eps)
        return np.array([[(dx_x - dx0) / eps, (dx_y - dx0) / eps], [(dy_x - dy0) / eps, (dy_y - dy0) / eps]])

    def _classify_fp(self, fp):
        J = self._get_jacobian(fp[0], fp[1])
        try:
            eigvals = np.linalg.eigvals(J)
            v1, v2 = eigvals[0], eigvals[1]
            if np.iscomplexobj(eigvals) or np.iscomplex(v1):
                if np.isclose(v1.real, 0, atol=1e-3): return "center"
                return "stable spiral" if v1.real < 0 else "unstable spiral"
            else:
                v1, v2 = v1.real, v2.real
                if np.isclose(v1, 0, atol=1e-3) or np.isclose(v2, 0, atol=1e-3): return "degenerate"
                if v1 * v2 < 0: return "saddle(unstable)"
                return "stable/sink" if v1 < 0 else "unstable/source"
        except:
            return "unknown"

    def find_fixed_points(self, tolerance=1e-4):
        fixed_points = []
        guess_x = np.linspace(self.x_range[0], self.x_range[1], 15)
        guess_y = np.linspace(self.y_range[0], self.y_range[1], 15)
        for gx in guess_x:
            for gy in guess_y:
                sol, _, ier, _ = fsolve(lambda v: self.system(v[0], v[1]), [gx, gy], full_output=True)
                if ier == 1 and (self.x_range[0] <= sol[0] <= self.x_range[1]) and (
                        self.y_range[0] <= sol[1] <= self.y_range[1]):
                    if not any(np.linalg.norm(sol - fp) < tolerance for fp in fixed_points):
                        fixed_points.append(sol)
        return fixed_points

    def plot(self, initial_conditions=None, t_span=(0, 20)):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        speed = np.sqrt(self.U ** 2 + self.V ** 2)
        ax.streamplot(self.X, self.Y, self.U, self.V, color=speed, cmap='viridis', linewidth=1, density=1.5)
        ax.contour(self.X, self.Y, self.U, levels=[0], colors='red', linewidths=2, alpha=0.7)
        ax.plot([], [], color='red', label=f'x-nullcline: {self.dx_rhs} = 0')
        ax.contour(self.X, self.Y, self.V, levels=[0], colors='green', linewidths=2, alpha=0.7)
        ax.plot([], [], color='green', label=f'y-nullcline: {self.dy_rhs} = 0')

        fixed_points = self.find_fixed_points()
        for fp in fixed_points:
            fp_type = self._classify_fp(fp)
            ax.plot(fp[0], fp[1], 'ko', markersize=8, markeredgecolor='white',
                    label=f'{fp_type} ({fp[0]:.2f}, {fp[1]:.2f})')
            if fp_type == "saddle(unstable)":
                evals, evecs = np.linalg.eig(self._get_jacobian(fp[0], fp[1]))
                for i in range(2):
                    color = 'blue' if evals[i].real < 0 else 'crimson'
                    ax.axline((fp[0], fp[1]), (fp[0] + evecs[0, i].real, fp[1] + evecs[1, i].real),
                              color=color, linestyle='--', alpha=0.6,
                              label="stable manifold" if evals[i].real < 0 else "unstable manifold")

        if initial_conditions:
            colors = plt.cm.tab10.colors
            for idx, ic in enumerate(initial_conditions):
                c = colors[idx % len(colors)]
                sol = solve_ivp(lambda t, s: self.system(s[0], s[1]), t_span, ic, max_step=0.05)
                ax.plot(sol.y[0], sol.y[1], color=c, linewidth=2.5)
                ax.plot(ic[0], ic[1], 's', color=c, markersize=6, label=f'IC: ({ic[0]:.2f}, {ic[1]:.2f})')

        ax.set_xlim(self.x_range);
        ax.set_ylim(self.y_range)
        ax.set_xlabel('x');
        ax.set_ylabel('y');
        ax.set_title(self.title)

        h, l = ax.get_legend_handles_labels()
        by_l = dict(zip(l, h))

        non_fp_keys = []
        fp_keys = []
        for k in by_l.keys():
            if any(keyword in k.lower() for keyword in ['nullcline', 'manifold', 'trajector', 'ic:']):
                non_fp_keys.append(k)
            else:
                fp_keys.append(k)

        sorted_keys = non_fp_keys + fp_keys
        sorted_handles = [by_l[k] for k in sorted_keys]

        ax.legend(sorted_handles, sorted_keys, loc='upper left', bbox_to_anchor=(1.01, 1.0))

        ax.grid(True, linestyle='--', alpha=0.6)
        plt.subplots_adjust(right=0.72)

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase_portrait_{current_time}.png"

        # 确保图例不会被剪掉
        plt.savefig(filename, bbox_inches='tight')
        print(f"😋图像已成功保存至项目文件夹: {filename}")

        # 关闭图表释放内存
        plt.close(fig)


def your_linear_system(x, y):
    a = 0.5
    dx = y
    dy = a-np.sin(x)
    return dx, dy


if __name__ == "__main__":
    pp = PhasePortrait2D(your_linear_system, title="♪───*\(>^ω^<)/*────♪")
    pp.plot(initial_conditions=[[0.0, 0.0]], t_span=(0, 5))