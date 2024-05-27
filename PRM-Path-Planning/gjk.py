import pygame
import sys
import math

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other):
        return Vec2(self.x * other, self.y * other)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def LengthSquared(self):
        return self.x**2 + self.y**2

    def Length(self):
        return math.sqrt(self.LengthSquared())

    def normalized(self):
        length = self.Length()
        if length > 0:
            return Vec2(self.x / length, self.y / length)
        return Vec2(0, 0)
# 定义长方形类
class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_points(self):
        return [
            Vec2(self.x, self.y),
            Vec2(self.x + self.width, self.y),
            Vec2(self.x + self.width, self.y + self.height),
            Vec2(self.x, self.y + self.height)
        ]

# 计算 Minkowski 差
def minkowski_difference(rect1, rect2):
    points1 = rect1.get_points()
    points2 = rect2.get_points()
    diff_points = []

    for p1 in points1:
        for p2 in points2:
            diff_points.append(p1 - p2)

    return diff_points

# GJK 算法
def gjk_algorithm(rect1, rect2):
    # 初始化搜索方向
    direction = Vec2(1, 0)

    # 初始化起始点
    support = support_func(rect1, rect2, direction)

    # 添加起始点到 simplex 中
    simplex = [support]

    # 开始迭代
    while True:
        # 反向搜索方向
        direction = Vec2(-simplex[-1].x, -simplex[-1].y)

        # 获取新的支撑点
        support = support_func(rect1, rect2, direction)

        # 如果新的支撑点没有向原点前进，则两个矩形重叠
        if support.dot(direction) < 0:
            return 0

        # 将新的支撑点添加到 simplex 中
        simplex.append(support)

        # 检查原点是否在 simplex 内
        if contains_origin(simplex):
            return compute_distance(simplex)

# 支撑函数
def support_func(rect1, rect2, direction):
    diff_points = minkowski_difference(rect1, rect2)
    farthest_point = None
    max_dot = -float('inf')

    for point in diff_points:
        dot_product = point.x * direction.x + point.y * direction.y
        if dot_product > max_dot:
            max_dot = dot_product
            farthest_point = point

    return farthest_point

# 判断原点是否在 simplex 内
def contains_origin(simplex):
    a = simplex[-1]
    ao = Vec2(-a.x, -a.y)

    if len(simplex) == 2:
        b = simplex[-2]
        ab = b - a
        ab_ao = ab.dot(ao)

        if ab_ao > 0:
            return False

        direction = ab.normalized().cross(Vec2(0, 0) - b)
        return ab.dot(ao) >= 0

    if len(simplex) == 3:
        b = simplex[-2]
        c = simplex[-3]
        ab = b - a
        ac = c - a

        ab_ao = ab.dot(ao)
        ac_ao = ac.dot(ao)

        if ab_ao > 0:
            simplex.remove(c)
            return False
        elif ac_ao > 0:
            simplex.remove(b)
            return False

        ab_cross_ac = ab.cross(ac)
        if ab_cross_ac * ab_ao <= 0:
            simplex.remove(c)
            return False

        if ab_cross_ac * ac_ao <= 0:
            simplex.remove(b)
            return False

        return True

# 计算最短距离
def compute_distance(simplex):
    a = simplex[-1]
    ao = Vec2(-a.x, -a.y)

    if len(simplex) == 2:
        b = simplex[-2]
        ab = b - a
        ab_ao = ab.dot(ao)

        return ab_ao / ab.length()

    if len(simplex) == 3:
        b = simplex[-2]
        c = simplex[-3]
        ab = b - a
        ac = c - a

        ab_cross_ac = ab.cross(ac)

        if ab_cross_ac > 0:
            return 0

        ab_ao = ab.dot(ao)
        ac_ao = ac.dot(ao)

        if ab_ao * ac_ao > 0:
            return 0

        if ab_ao * ac_ao < 0:
            return abs(ab_cross_ac) / ac.length()

        return 0

# 绘制矩形
def draw_rectangle(screen, rect, color):
    pygame.draw.rect(screen, color, (rect.x, rect.y, rect.width, rect.height))

# 主函数
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption('GJK Algorithm Example')
    clock = pygame.time.Clock()

    rect1 = Rectangle(100, 100, 100, 100)
    rect2 = Rectangle(300, 300, 150, 50)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        draw_rectangle(screen, rect1, (0, 0, 255))
        draw_rectangle(screen, rect2, (255, 0, 0))

        distance = gjk_algorithm(rect1, rect2)
        print("Distance between rectangles:", distance)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
