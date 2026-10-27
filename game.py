import random
import cv2
import numpy as np

W, H = 960, 540


def load_img(path, width):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)),
                      interpolation=cv2.INTER_AREA)


def load_background(path):
    bg = cv2.imread(path)
    if bg is None:
        raise FileNotFoundError(f"Background tidak ditemukan: {path}")
    bh, bw = bg.shape[:2]
    target_h = int(bw * H / W)
    if target_h <= bh:
        top = (bh - target_h) // 2
        bg = bg[top:top + target_h, :]
    else:
        target_w = int(bh * W / H)
        left = (bw - target_w) // 2
        bg = bg[:, left:left + target_w]
    return cv2.resize(bg, (W, H))


def overlay(bg, img, x, y):
    x, y = int(x), int(y)
    h, w = img.shape[:2]
    if x >= bg.shape[1] or y >= bg.shape[0] or x + w <= 0 or y + h <= 0:
        return
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])
    crop = img[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = crop[:, :, 3:4] / 255.0
    region = bg[y1:y2, x1:x2]
    bg[y1:y2, x1:x2] = (alpha * crop[:, :, :3] +
                        (1 - alpha) * region).astype(np.uint8)


def hit(a, b, shrink=0.25):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax += aw * shrink; ay += ah * shrink; aw *= (1 - 2 * shrink); ah *= (1 - 2 * shrink)
    bx += bw * shrink; by += bh * shrink; bw *= (1 - 2 * shrink); bh *= (1 - 2 * shrink)
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class Player:
    def __init__(self, img):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.x = 80.0
        self.y = H / 2 - self.h / 2
        self.speed = 6

    def update(self, state, target_y):
        if state == "white":
            self.x += self.speed   
        elif state == "black":
            self.x -= self.speed      
        if target_y is not None:    
            self.y += (target_y - self.h / 2 - self.y) * 0.15
        self.x = max(0, min(W - self.w, self.x))
        self.y = max(0, min(H - self.h, self.y))

    def rect(self):
        return (self.x, self.y, self.w, self.h)

    def draw(self, canvas):
        overlay(canvas, self.img, self.x, self.y)


class Obstacle:
    def __init__(self, img, speed):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.x = float(W)
        self.y = random.randint(0, H - self.h)
        self.speed = speed

    def update(self):
        self.x -= self.speed

    def off_screen(self):
        return self.x + self.w < 0

    def rect(self):
        return (self.x, self.y, self.w, self.h)

    def draw(self, canvas):
        overlay(canvas, self.img, self.x, self.y)


class Game:
    def __init__(self):
        self.bg = load_background("assets/background.png")
        self.player_img = load_img("assets/mc.png", 130)
        self.zaku_imgs = [load_img("assets/zaku 1.png", 110),
                          load_img("assets/zaku 2.webp", 110)]
        self.high_score = 0
        self.reset()

    def reset(self):
        self.player = Player(self.player_img)
        self.obstacles = []
        self.lives = 3
        self.score = 0
        self.level = 1
        self.frame = 0
        self.invincible = 0
        self.game_over = False
        self.bg_offset = 0

    def update(self, state, target_y):
        if self.game_over:
            return
        self.frame += 1
        self.bg_offset = (self.bg_offset + 2 + self.level) % W

        self.player.update(state, target_y)

        spawn_gap = max(25, 70 - self.level * 6)
        if self.frame % spawn_gap == 0:
            img = random.choice(self.zaku_imgs)
            self.obstacles.append(Obstacle(img, random.randint(4, 7) + self.level))

        for o in self.obstacles:
            o.update()
            if self.invincible == 0 and hit(self.player.rect(), o.rect()):
                self.lives -= 1
                self.invincible = 60
                o.x = -999
        self.obstacles = [o for o in self.obstacles if not o.off_screen()]

        if self.invincible > 0:
            self.invincible -= 1

        self.score += 1
        self.level = 1 + self.score // 600

        if self.lives <= 0:
            self.game_over = True
            self.high_score = max(self.high_score, self.score)

    def draw(self, paused=False):
        canvas = np.roll(self.bg, -self.bg_offset, axis=1).copy()
        for o in self.obstacles:
            o.draw(canvas)
        if self.invincible == 0 or (self.invincible // 5) % 2 == 0:
            self.player.draw(canvas)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Score: {self.score}   Level: {self.level}",
                    (15, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Nyawa: {self.lives}   High: {self.high_score}",
                    (15, 60), font, 0.7, (0, 255, 255), 2)
        if paused:
            cv2.putText(canvas, "PAUSE", (W // 2 - 70, H // 2), font, 1.5, (255, 255, 255), 3)
        if self.game_over:
            cv2.putText(canvas, "GAME OVER", (W // 2 - 160, H // 2 - 10), font, 1.8, (0, 0, 255), 4)
            cv2.putText(canvas, "Tekan R untuk restart", (W // 2 - 190, H // 2 + 40),
                        font, 0.9, (255, 255, 255), 2)
        return canvas