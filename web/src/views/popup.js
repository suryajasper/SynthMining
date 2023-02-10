import m from 'mithril';
import '../css/popup.scss';
import { icons } from './icons';

const popupStatus = {
  active: false,
}

function updatePopupOverlayStatus({active}) {
  popupStatus.active = active;
}

window.onkeydown = e => {
  if (e.key === 'Escape') {
    updatePopupOverlayStatus({active: false});
    m.redraw();
  }
}

function PopupOverlay() {
  let prevActive = popupStatus.active;

  return {
    view(vnode) {
      if (prevActive !== popupStatus.active && !popupStatus.active)
        vnode.attrs.disabledCallback();
      prevActive = popupStatus.active;

      return m('div.popup-background-blur', {
        class: popupStatus.active ? 'enabled' : '',
      },
        vnode.children, 
      );
    }
  };
}

const PopupOuter = {

  view(vnode) {
    
  }
}

class Popup {
  constructor() {
    this.input = {};
    this.active = false;
    this.title = '';
    this.actions = [];
  }

  createInputGroup({ 
    id, 
    displayTitle=id, 
    type='text', 
    placeholder='', 
    initialValue='',
  }) {
    let inputSelector = type === 'textarea' ? 'textarea.input-group-textarea' : 'input.input-group-input';
    let inputAttrs = {
      name: id, placeholder,
      value: this.input[id] !== undefined ? this.input[id] : initialValue,
      oninput: e => {
        this.input[id] = e.target.value;
      },
    };
    if (type !== 'textarea') inputAttrs.type = type;

    return m('div.input-group', {
      class: type === 'textarea' ? 'long-in' : '',
    }, [
      m('label.input-group-label', { for: id }, displayTitle),

      m(inputSelector, inputAttrs),
    ]);
  }

  createCallbackButton(action) {
    return m('button.popup-button', {
      onclick: e => {
        action.res(e);
      }
    },
      action.name, 
    )
  }

  hidePopup(vnode) {
    this.input = {};
    this.active = false;
    vnode.attrs.disabledCallback();
  }

  loadPopupContent(vnode) { return null; }

  view(vnode) {
    return m('div.popup-container', {
      style: { display: vnode.attrs.active ? 'flex' : 'none', },
      tabindex: 0,
      // onblur: e => this.hidePopup(vnode),
    }, [
      m('div.popup-header', [
        m('span.popup-title', this.title),

        m('button.exit-view-button', {
          title: 'Cancel',
          onclick: e => { this.hidePopup(vnode); }
        },
          icons.exit,
        )
      ]),
      m('div.popup-content', 
        this.loadPopupContent(vnode),
      ),
      m('div.popup-footer', 
        this.actions
          .map(this.createCallbackButton)
      ),
    ]);
  }
}

export { updatePopupOverlayStatus, PopupOverlay, Popup };