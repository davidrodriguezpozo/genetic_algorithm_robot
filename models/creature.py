from enum import Enum
from typing import List
from xml.dom.minidom import getDOMImplementation

import numpy as np
from models import genome
from models.genome import URDFLink


class MotorType(Enum):
    PULSE = 1
    SINE = 2


class Motor:
    def __init__(self, contorl_waveform, control_amp, control_freq):
        if contorl_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = control_amp
        self.freq = control_freq
        self.phase = 0

    def get_output(self):
        output = 0
        self.phase = (self.phase + self.freq) % (np.pi * 2)
        if self.motor_type == MotorType.PULSE:
            if self.phase < np.pi:
                output = 1
            else:
                output = -1
        elif self.motor_type == MotorType.SINE:
            output = np.sin(self.phase)
        return output


class Creature:
    def __init__(self, gene_count):
        self.spec = genome.GenomeGenerator.get_gene_spec()
        self.dna = genome.GenomeGenerator.get_random_genome(len(self.spec), gene_count)
        self.flat_links: List[URDFLink] = []
        self.expanded_links: List[URDFLink] = []
        self.motors: List[Motor] = None
        self.get_flat_links()
        self.get_expanded_links()
        self.start_position = None
        self.last_position = None
        self.dist = 0

    def update_dna(self, dna):
        self.dna = dna
        self.reset()

    def reset(self):
        self.flat_links: List[URDFLink] = []
        self.expanded_links: List[URDFLink] = []
        self.motors: List[Motor] = None
        self.start_position = None
        self.last_position = None
        self.dist = 0

    def get_flat_links(self):
        gdicts = genome.GenomeGenerator.get_genome_dicts(self.dna, self.spec)
        self.flat_links = genome.GenomeGenerator.genome_to_links(gdicts)
        return self.flat_links

    def get_expanded_links(self):
        self.get_flat_links()
        exp_links = [self.flat_links[0]]
        genome.GenomeGenerator.expand_links(
            self.flat_links, self.flat_links[0], self.flat_links[0].name, exp_links
        )
        self.expanded_links = exp_links
        return self.expanded_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        for link in self.expanded_links:
            robot_tag.appendChild(link.to_link_element(adom))
        first = True
        for link in self.expanded_links:
            if first:  # skip the root node!
                first = False
                continue
            robot_tag.appendChild(link.to_joint_element(adom))
            robot_tag.setAttribute("name", "pepe")  # choose a name!
        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        motors = []
        if self.motors is not None:
            return self.motors
        for i in range(1, len(self.expanded_links)):
            l = self.expanded_links[i]
            m = Motor(l.control_waveform, l.control_amp, l.control_freq)
            motors.append(m)
        self.motors = motors
        return self.motors

    def update_position(self, pos):
        if self.last_position is not None:
            p1 = np.array(self.last_position)
            p2 = np.array(pos)
            dist = np.linalg.norm(p1 - p2)
            self.dist = self.dist + dist

        self.last_position = pos

    def get_distance_travelled(self):
        return self.dist
