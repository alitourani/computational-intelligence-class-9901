import pygame
import neat
import time
import os
import random

pygame.font.init()

WinHeight = 700
WinWidth = 510

GEN = 0

ST_FONT = pygame.font.SysFont("comicsans", 50)

# axe parande ha vared krde va andeze an ha ro do barabar mikonim
BIRD_PNG = pygame.transform.scale2x(pygame.image.load(
    os.path.join("imgs", "bird.png")))
# axe sotun
PIPE_PNG = [pygame.transform.scale2x(pygame.image.load(
    os.path.join("imgs", "pipeBottom.png"))),
    pygame.transform.scale2x(pygame.image.load(
        os.path.join("imgs", "pipeTop.png")))]
# axe pass zamine
BG_PNG = pygame.transform.scale2x(pygame.image.load(
    os.path.join("imgs", "bg.png")))


# objecte bird
class Bird:
    IMG = BIRD_PNG
    ROTATION_VELOCITY = 20

    def __init__(self, x, y):
        # noghte shurue parande
        self.x = x
        self.y = y
        self.tick_counter = 0
        self.velocity = 0
        self.height = 0
        self.img = self.IMG

    # paridane parande
    def jump(self):
        # mizane bala raftn
        self.velocity = -10.5
        # akharin paresh
        self.tick_counter = 0
        self.height = self.y

    def move(self):
        # ye frame jolotar rftim
        self.tick_counter += 1
        # chand pixel bala ya paen rftim
        d = self.velocity * self.tick_counter + 1.5 * self.tick_counter ** 2
        # bishtr az 16 pixel paen nare
        if d >= 16:
            d = 16

        # boland tar mipre
        if d < 0:
            d -= 2

        self.y = self.y + d

    # mikeshe parande ro
    def draw(self, win):
        self.img = self.IMG
        win.blit(self.img, (self.x, self.y))

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

 
class Pipe:
    # fasele do sotun
    GAP = 200
    # sorate harekate sotunha
    VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.TOP_PIPE = PIPE_PNG[1]
        self.BOTTOM_PIPE = PIPE_PNG[0]
        self.passed = False
        self.set_height()

    # andaze randome ertefae sotun ha
    def set_height(self):
        self.height = random.randrange(50, 390)
        self.top = self.height - self.TOP_PIPE.get_height()
        self.bottom = self.height + self.GAP

    # ye meghdar sotunbe samte chap ta miad
    def move(self):
        self.x -= self.VELOCITY

    # keshidane sotun ha
    def draw(self, win):
        win.blit(self.TOP_PIPE, (self.x, self.top))
        win.blit(self.BOTTOM_PIPE, (self.x, self.bottom))

    # barresie barkhord kardan parande be sotun
    def collide(self, bird):
        birdMask = bird.get_mask()
        topMask = pygame.mask.from_surface(self.TOP_PIPE)
        bottomMask = pygame.mask.from_surface(self.BOTTOM_PIPE)

        topOffset = (self.x - bird.x, self.top - round(bird.y))
        bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))

        # agar overlap nadashte bashad None barmigrdanad
        b_point = birdMask.overlap(bottomMask, bottomOffset)
        t_point = birdMask.overlap(topMask, topOffset)

        # agar harkodum None nabudn yani brkhord dashtim
        if t_point or b_point:
            return True

        return False


def draw_window(win, birds, pipes, score, gen, pipe_ind):
    win.blit(BG_PNG, (0, -200))
    # keshidne pipe ha
    for pipe in pipes:
        pipe.draw(win)

    # neveshtane score
    txt = ST_FONT.render("Score: " + str(score), 1, (128, 0, 0))
    win.blit(txt, (WinWidth - 10 - txt.get_width(), 10))

    # neveshtane gen
    txt = ST_FONT.render("Gen: " + str(gen), 1, (0, 255, 255))
    win.blit(txt, (10, 10))

    for bird in birds:
        try:
            pygame.draw.line(win, (255, 0, 0), (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                             (pipes[pipe_ind].x + pipes[pipe_ind].TOP_PIPE.get_width() / 2, pipes[pipe_ind].height), 5)
            pygame.draw.line(win, (255, 0, 0), (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                             (pipes[pipe_ind].x + pipes[pipe_ind].BOTTOM_PIPE.get_width() / 2, pipes[pipe_ind].bottom),
                             5)
        except:
            pass
        bird.draw(win)
    pygame.display.update()


# fitness function
def fitness(genomes, config):
    global GEN
    GEN += 1

    pygame.init()
    birds = []
    Nets = []
    ge = []

    # barae har parande Net va genom drnazar gereftim 
    # genom ha touple hastand va ma faghar object gen ro mikhiam 
    for _, g in genomes:
        # set krdne shabake asabi
        Net = neat.nn.FeedForwardNetwork.create(g, config)
        Nets.append(Net)
        # be vojud avardane parande ,noghte shurue parande
        birds.append(Bird(230, 350))
        # genash ra ba fitness 0 varede list mikonim
        g.fitness = 0
        ge.append(g)

    # position pipeha
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WinWidth, WinHeight))
    clock = pygame.time.Clock()
    score = 0
    run = True

    while run:
        clock.tick(30)

        if score == 50:
            print('**********\nwe achieved the goal\n**********')
            run = False
            pygame.quit()
            quit()

        # barae kharej shodn az pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # agar bishtar az ye pipe dashte
        # bashim va az un lule rad shodim rue pipe badi tamarkoz kone
        # az anjai k mogheiate x parande ha bhm brabare mohem nist kodum parande ro danazar begirim
        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].TOP_PIPE.get_width():
                pipe_index = 1
        # agar parandei baghi namunde bud az bazi kharej mishe
        else:
            run = False
            break
        # harekate parande ha va ...
        for x, bird in enumerate(birds):
            bird.move()
            # har bar jolo raft behesh fitness midim ta bishtr zende bemune 
            # chon in function 30 bar dar sanie ejra mishe fitnessesho kamtar ddm
            ge[x].fitness += 0.1

            # bbinim bepare ya na
            output = Nets[x].activate((bird.y, abs(
                bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        # sotun hae remove shode
        rem = []

        # harekate sotun ha va chek krdne barkhord ba parande ha
        for pipe in pipes:
            for x, bird in enumerate(birds):
                # har bar k parande be yek sotun barkhord kond az fitnesash kam mishavad
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    # hazfe parande
                    birds.pop(x)
                    Nets.pop(x)
                    ge.pop(x)
                # chek mikonim bbinim sotun ro rad krdim ya na
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            # age majmueshun kamtar az 0 beshe yani az screen kharej shode
            if pipe.x + pipe.TOP_PIPE.get_width() < 0:
                rem.append(pipe)
            # harekat sotun
            pipe.move()

        # age true bud be score ezafe mishe va pipe jadid ezafe mishe
        if add_pipe:
            score += 1
            # bade har bar hazfe parande ha har kodum k munde bud 5 ta be
            # fitnessesh ezafe mishe pas tashvigh mishan k az vasate pipe ha bishtr rad beshan
            if g in ge:
                g.fitness += 2
            pipes.append(Pipe(700))

        #  pak krdne sotun hae rad shode
        for r in rem:
            pipes.remove(r)

        # barresie in ke parande be zamin ya balatar az asemun barkhord krde ya na
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 700 or bird.y < 0:
                # hazfe parande
                birds.pop(x)
                Nets.pop(x)
                ge.pop(x)

        draw_window(win, birds, pipes, score, GEN, pipe_index)


def run(config_path):
    # peida krdne sartitr hae lazem
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # tolide population
    population = neat.Population(config)

    # 50 bar fitness k naghshe fitness function
    # ra dare run mikone va me gen ha ro be fitness mide
    winner = population.run(fitness, 50)


# add kardane configuration
if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
