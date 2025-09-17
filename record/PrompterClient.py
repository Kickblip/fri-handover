import pygame

class PrompterClient:
    def __init__(self, title="Prompter", size=(720, 220), font_size=64):
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode(size)
        self.font = pygame.font.SysFont(None, font_size)
        self.text = ""
        self.bg = (34, 34, 34)
        self.fg = (255, 255, 255)
        self.alive = True
        self._draw()

    def _pump(self):
        if not self.alive:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.alive = False

    def show(self, text, bg=None):
        if not self.alive:
            return
        self.text = str(text)
        if bg is not None:
            self.bg = bg
        self._pump()
        self._draw()

    def _draw(self):
        if not self.alive:
            return
        self.screen.fill(self.bg)
        if self.text:
            surf = self.font.render(self.text, True, self.fg)
            rect = surf.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(surf, rect)
        pygame.display.flip()

    def close(self):
        if self.alive:
            self.alive = False
            pygame.quit()
