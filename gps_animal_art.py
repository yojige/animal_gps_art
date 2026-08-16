"""
gps_animal_art.py

画像
 ↓
輪郭抽出
 ↓
新宿区周辺へ配置
 ↓
OpenStreetMap道路ネットワーク取得
 ↓
輪郭ポイントを最近傍道路ノードへスナップ
 ↓
道路ネットワーク上の最短経路で接続
 ↓
約10kmになる縮尺を探索
 ↓
GPX出力
 ↓
PNGプレビュー出力

使用例:

    python gps_animal_art.py dog.png

または

    python gps_animal_art.py dog.png --target-km 10

"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import osmnx as ox
import networkx as nx
import gpxpy
import gpxpy.gpx

from shapely.geometry import LineString
import matplotlib.pyplot as plt


# ============================================================
# 設定
# ============================================================

DEFAULT_PLACE = "shinjuku, Tokyo, Japan"

DEFAULT_TARGET_KM = 10.0

# 動物画像を地図上に配置する際の初期幅
DEFAULT_WIDTH_KM = 5.0

# 輪郭の点数
DEFAULT_CONTOUR_POINTS = 100

# 画像の余白
IMAGE_MARGIN = 20


# ============================================================
# 画像から輪郭を抽出
# ============================================================

def extract_contour(
    image_path,
    num_points=DEFAULT_CONTOUR_POINTS
):
    """
    画像から最大輪郭を抽出し、
    -1～+1程度の正規化座標として返す。

    戻り値:
        ndarray shape=(N,2)
        x,y
    """

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"画像を読み込めません: {image_path}"
        )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # 自動二値化
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 小さなノイズを除去
    kernel = np.ones((3, 3), np.uint8)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise RuntimeError(
            "画像から輪郭を検出できませんでした。"
        )

    # 最大面積の輪郭
    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    if area < 100:
        raise RuntimeError(
            "検出された輪郭が小さすぎます。"
        )

    contour = contour[:, 0, :].astype(float)

    # 輪郭を均等な距離でサンプリング
    contour = resample_closed_curve(
        contour,
        num_points
    )

    # x/yを0～1へ正規化
    min_xy = contour.min(axis=0)
    max_xy = contour.max(axis=0)

    size = max_xy - min_xy

    # ゼロ除算対策
    size[size == 0] = 1

    normalized = (
        contour - min_xy
    ) / size

    return normalized


# ============================================================
# 閉曲線を均等サンプリング
# ============================================================

def resample_closed_curve(points, n):
    """
    閉じた輪郭を等距離に近い形でn点にする。
    """

    points2 = np.vstack([
        points,
        points[0]
    ])

    segments = np.diff(
        points2,
        axis=0
    )

    lengths = np.linalg.norm(
        segments,
        axis=1
    )

    cumulative = np.concatenate([
        [0],
        np.cumsum(lengths)
    ])

    total = cumulative[-1]

    if total == 0:
        return points[:n]

    distances = np.linspace(
        0,
        total,
        n,
        endpoint=False
    )

    result = []

    for d in distances:

        idx = np.searchsorted(
            cumulative,
            d,
            side="right"
        ) - 1

        idx = min(
            idx,
            len(segments) - 1
        )

        local = (
            d - cumulative[idx]
        ) / lengths[idx]

        p = (
            points2[idx]
            + local * segments[idx]
        )

        result.append(p)

    return np.array(result)


# ============================================================
# OSM道路ネットワーク取得
# ============================================================

def download_road_network(place):
    """
    OpenStreetMapから道路ネットワークを取得。
    """

    print("OpenStreetMapから道路データを取得しています...")
    print(f"対象地域: {place}")

    # 自転車・徒歩で利用可能な道路を中心に取得
    G = ox.graph.graph_from_place(
        place,
        network_type="bike",
        simplify=True,
        retain_all=False
    )

    print(
        f"取得したノード数: {len(G.nodes):,}"
    )

    print(
        f"取得したエッジ数: {len(G.edges):,}"
    )

    return G


# ============================================================
# グラフを投影
# ============================================================

def prepare_graph(G):

    # 日本の地域で距離計算をするため、
    # メートル系の投影座標へ変換
    G_proj = ox.projection.project_graph(G)

    return G_proj


# ============================================================
# グラフ中心を取得
# ============================================================

def graph_center(G):

    nodes, edges = ox.convert.graph_to_gdfs(
        G
    )

    center = nodes.geometry.unary_union.centroid

    return center.x, center.y


# ============================================================
# 画像輪郭を地図座標へ変換
# ============================================================

def contour_to_map(
    contour,
    center_x,
    center_y,
    width_m
):
    """
    正規化された輪郭を、
    地図上のメートル座標へ変換する。

    縦横比は画像を維持する。
    """

    x = (
        contour[:, 0] - 0.5
    ) * width_m

    # 画像座標はY下向きなので反転
    y = (
        0.5 - contour[:, 1]
    ) * width_m

    return np.column_stack([
        center_x + x,
        center_y + y
    ])


# ============================================================
# 最近傍道路ノードへスナップ
# ============================================================

def snap_contour_to_roads(
    G,
    points
):
    """
    輪郭ポイントを最近傍道路ノードへスナップ。
    """

    X = points[:, 0]
    Y = points[:, 1]

    nodes, distances = (
        ox.distance.nearest_nodes(
            G,
            X,
            Y,
            return_dist=True
        )
    )

    return nodes, distances


# ============================================================
# 重複ノードを除去
# ============================================================

def remove_duplicate_nodes(nodes):

    result = []

    previous = None

    for node in nodes:

        node = int(node)

        if node != previous:
            result.append(node)

        previous = node

    return result


# ============================================================
# 道路ネットワーク上で輪郭を接続
# ============================================================

def route_nodes(
    G,
    contour_nodes
):
    """
    輪郭上の道路ノードを順番に
    最短経路で接続する。
    """

    route = []

    for i in range(len(contour_nodes)):

        start = int(
            contour_nodes[i]
        )

        end = int(
            contour_nodes[
                (i + 1) % len(contour_nodes)
            ]
        )

        if start == end:
            continue

        try:

            path = nx.shortest_path(
                G,
                start,
                end,
                weight="length"
            )

        except nx.NetworkXNoPath:

            print(
                f"経路がありません: "
                f"{start} -> {end}"
            )

            continue

        if route:
            route.extend(path[1:])
        else:
            route.extend(path)

    return route


# ============================================================
# 道路ルートから座標列を取得
# ============================================================

def route_to_coordinates(
    G,
    route
):

    coords = []

    for node in route:

        data = G.nodes[node]

        coords.append(
            (
                float(data["x"]),
                float(data["y"])
            )
        )

    return np.array(coords)


# ============================================================
# ルート距離
# ============================================================

def route_distance_m(G, route):

    total = 0.0

    for u, v in zip(
        route[:-1],
        route[1:]
    ):

        data = G.get_edge_data(
            u,
            v
        )

        if data is None:
            continue

        # MultiDiGraphなので
        # 最短のedgeを使う
        length = min(
            float(edge_data.get(
                "length",
                0
            ))
            for edge_data in data.values()
        )

        total += length

    return total


# ============================================================
# GPSアート生成
# ============================================================

def generate_route(
    G,
    contour,
    width_km
):

    nodes, edges = ox.convert.graph_to_gdfs(
        G
    )

    center = nodes.geometry.unary_union.centroid

    center_x = center.x
    center_y = center.y

    width_m = width_km * 1000

    map_points = contour_to_map(
        contour,
        center_x,
        center_y,
        width_m
    )

    snapped_nodes, snap_distances = (
        snap_contour_to_roads(
            G,
            map_points
        )
    )

    snapped_nodes = remove_duplicate_nodes(
        snapped_nodes
    )

    if len(snapped_nodes) < 5:
        raise RuntimeError(
            "道路へスナップした結果、"
            "ノード数が少なすぎます。"
        )

    route = route_nodes(
        G,
        snapped_nodes
    )

    distance = route_distance_m(
        G,
        route
    )

    return (
        route,
        distance,
        map_points,
        snapped_nodes,
        snap_distances
    )


# ============================================================
# 10kmに最も近い縮尺を探索
# ============================================================

def find_best_scale(
    G,
    contour,
    target_km
):

    print()
    print(
        f"目標距離: {target_km:.2f} km"
    )

    # 最初に広めの候補を試す
    candidates = np.linspace(
        2.0,
        8.0,
        13
    )

    results = []

    for width_km in candidates:

        try:

            (
                route,
                distance,
                map_points,
                snapped_nodes,
                snap_distances
            ) = generate_route(
                G,
                contour,
                width_km
            )

            distance_km = (
                distance / 1000
            )

            error = abs(
                distance_km - target_km
            )

            results.append({
                "width_km": width_km,
                "distance_km": distance_km,
                "error": error,
                "route": route,
                "map_points": map_points,
                "snapped_nodes": snapped_nodes,
                "snap_distances": snap_distances
            })

            print(
                f"幅 {width_km:.2f} km"
                f" → {distance_km:.2f} km"
            )

        except Exception as e:

            print(
                f"幅 {width_km:.2f} km:"
                f" 失敗 {e}"
            )

    if not results:
        raise RuntimeError(
            "有効なGPSルートを作成できませんでした。"
        )

    best = min(
        results,
        key=lambda x: x["error"]
    )

    # --------------------------------------------------------
    # さらに細かく探索
    # --------------------------------------------------------

    center_width = best["width_km"]

    fine_candidates = np.linspace(
        max(1.0, center_width - 0.6),
        center_width + 0.6,
        13
    )

    for width_km in fine_candidates:

        try:

            (
                route,
                distance,
                map_points,
                snapped_nodes,
                snap_distances
            ) = generate_route(
                G,
                contour,
                width_km
            )

            distance_km = (
                distance / 1000
            )

            error = abs(
                distance_km - target_km
            )

            results.append({
                "width_km": width_km,
                "distance_km": distance_km,
                "error": error,
                "route": route,
                "map_points": map_points,
                "snapped_nodes": snapped_nodes,
                "snap_distances": snap_distances
            })

        except Exception:
            pass

    best = min(
        results,
        key=lambda x: x["error"]
    )

    print()
    print(
        "========== BEST =========="
    )

    print(
        f"地図上の幅: "
        f"{best['width_km']:.2f} km"
    )

    print(
        f"実道路距離: "
        f"{best['distance_km']:.2f} km"
    )

    print(
        f"誤差: "
        f"{best['error']:.2f} km"
    )

    print(
        "=========================="
    )

    return best


# ============================================================
# GPX生成
# ============================================================

def save_gpx(
    G,
    route,
    filename
):

    gpx = gpxpy.gpx.GPX()

    gpx.name = "Animal GPS Art"

    track = gpxpy.gpx.GPXTrack()

    track.name = "Animal GPS Art"

    gpx.tracks.append(track)

    segment = gpxpy.gpx.GPXTrackSegment()

    track.segments.append(segment)

    for node in route:

        data = G.nodes[node]

        # Gのx = longitude
        # Gのy = latitude
        #
        # project_graph後はメートル座標なので
        # 元のEPSG:4326へ戻す

    # --------------------------------------------------------
    # 投影を戻す
    # --------------------------------------------------------

    G_ll = ox.projection.project_graph(
        G,
        to_crs="EPSG:4326"
    )

    for node in route:

        data = G_ll.nodes[node]

        point = gpxpy.gpx.GPXTrackPoint(
            latitude=float(data["y"]),
            longitude=float(data["x"])
        )

        segment.points.append(point)

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            gpx.to_xml()
        )

    print(
        f"GPXを書き出しました: {filename}"
    )


# ============================================================
# プレビュー画像
# ============================================================

def save_preview(
    G,
    route,
    filename
):

    G_ll = ox.projection.project_graph(
        G,
        to_crs="EPSG:4326"
    )

    fig, ax = ox.plot.plot_graph(
        G_ll,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.4,
        show=False,
        close=False
    )

    xs = []
    ys = []

    for node in route:

        data = G_ll.nodes[node]

        xs.append(
            float(data["x"])
        )

        ys.append(
            float(data["y"])
        )

    # 最後を閉じる
    if xs:
        xs.append(xs[0])
        ys.append(ys[0])

    ax.plot(
        xs,
        ys,
        linewidth=3
    )

    ax.set_title(
        "Animal GPS Art"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"プレビューを書き出しました: {filename}"
    )


# ============================================================
# メイン
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "画像から動物GPSアートを作成"
        )
    )

    parser.add_argument(
        "image",
        help="動物画像"
    )

    parser.add_argument(
        "--target-km",
        type=float,
        default=10.0,
        help="目標距離(km)"
    )

    parser.add_argument(
        "--place",
        default=DEFAULT_PLACE,
        help="対象地域"
    )

    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_CONTOUR_POINTS,
        help="輪郭ポイント数"
    )

    parser.add_argument(
        "--output",
        default="animal_gps_art.gpx",
        help="GPX出力ファイル"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. 輪郭抽出
    # --------------------------------------------------------

    print()
    print(
        "1. 動物画像から輪郭を抽出"
    )

    contour = extract_contour(
        args.image,
        args.points
    )

    print(
        f"輪郭ポイント数: {len(contour)}"
    )

    # --------------------------------------------------------
    # 2. OSM道路取得
    # --------------------------------------------------------

    print()
    print(
        "2. OpenStreetMap道路データ取得"
    )

    G = download_road_network(
        args.place
    )

    # --------------------------------------------------------
    # 3. 投影
    # --------------------------------------------------------

    print()
    print(
        "3. 距離計算用に座標変換"
    )

    G = prepare_graph(G)

    # --------------------------------------------------------
    # 4. 最適な縮尺探索
    # --------------------------------------------------------

    print()
    print(
        "4. 約10kmになる縮尺を探索"
    )

    best = find_best_scale(
        G,
        contour,
        args.target_km
    )

    route = best["route"]

    # --------------------------------------------------------
    # 5. GPX
    # --------------------------------------------------------

    print()
    print(
        "5. GPX生成"
    )

    save_gpx(
        G,
        route,
        args.output
    )

    # --------------------------------------------------------
    # 6. プレビュー
    # --------------------------------------------------------

    preview = os.path.splitext(
        args.output
    )[0] + "_preview.png"

    save_preview(
        G,
        route,
        preview
    )

    print()
    print(
        "処理完了"
    )


if __name__ == "__main__":
    main()